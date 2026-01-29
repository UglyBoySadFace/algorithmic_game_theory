#!/usr/bin/env python3

"""
Week 9: Counterfactual Regret Minimization (CFR) and CFR+

This module implements:
1. Standard Counterfactual Regret Minimization (CFR)
2. CFR+ variant with regret floor at 0
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from solutions.kuhn_poker import KuhnPokerNumpy as KuhnPoker
from solutions.kuhn_poker import State as KuhnPokerState
from solutions.week07.week07 import (
    GameNode, traverse_tree, InfosetKey, Strategy, StrategyProfile,
    evaluate, _format_infoset_for_print, collect_infosets
)


def _regret_matching(cumulative_regrets: np.ndarray) -> np.ndarray:
    """Convert cumulative regrets to a strategy using regret matching.
    
    Args:
        cumulative_regrets: Array of cumulative regrets for each action
    
    Returns:
        Strategy probabilities for each action
    """
    positive_regrets = np.maximum(cumulative_regrets, 0)
    regret_sum = np.sum(positive_regrets)
    
    if regret_sum > 0:
        return positive_regrets / regret_sum
    else:
        # Uniform strategy if no positive regrets
        return np.ones(len(cumulative_regrets)) / len(cumulative_regrets)


def _compute_counterfactual_regrets(
    node: GameNode,
    player: int,
    current_strategies: StrategyProfile,
    reach_probs: List[float]
) -> Tuple[float, Dict[InfosetKey, np.ndarray]]:
    """Compute counterfactual values and regrets for a player.
    
    Args:
        node: Current game node
        player: Player to compute regrets for (0 or 1)
        current_strategies: Current strategy profile for both players
        reach_probs: List of reach probabilities [prob_p0, prob_p1, prob_chance]
    
    Returns:
        Tuple of (counterfactual_value, regrets_dict)
        - counterfactual_value: Expected utility for player at this node
        - regrets_dict: Dict mapping infoset_key -> regret array for each action
    """
    if node.is_terminal:
        return node.rewards[player], {}
    
    if node.is_chance:
        # Chance node - weight by chance probabilities
        expected_value = 0.0
        all_regrets = {}
        
        for action in node.legal_actions:
            chance_prob = node.state.chance_strategy[action]
            child = node.children[action]
            new_reach = reach_probs.copy()
            new_reach[2] *= chance_prob
            
            child_value, child_regrets = _compute_counterfactual_regrets(
                child, player, current_strategies, new_reach
            )
            expected_value += chance_prob * child_value
            
            # Merge regrets from child
            for iset, regrets in child_regrets.items():
                if iset not in all_regrets:
                    all_regrets[iset] = regrets
                else:
                    all_regrets[iset] = all_regrets[iset] + regrets
        
        return expected_value, all_regrets
    
    # Player decision node
    current_player = node.player
    infoset_key = node.infoset_key
    
    # Get current strategy at this information set
    if infoset_key in current_strategies[current_player]:
        strategy = current_strategies[current_player][infoset_key]
    else:
        # Uniform if not specified
        num_actions = len(node.legal_actions)
        strategy = np.ones(num_actions) / num_actions
    
    # Compute value for each action
    action_values = np.zeros(len(node.legal_actions))
    all_regrets = {}
    
    for i, action in enumerate(node.legal_actions):
        child = node.children[action]
        
        # Update reach probabilities
        new_reach = reach_probs.copy()
        if current_player == 0:
            new_reach[0] *= strategy[i]
        else:
            new_reach[1] *= strategy[i]
        
        child_value, child_regrets = _compute_counterfactual_regrets(
            child, player, current_strategies, new_reach
        )
        action_values[i] = child_value
        
        # Merge regrets from child
        for iset, regrets in child_regrets.items():
            if iset not in all_regrets:
                all_regrets[iset] = regrets
            else:
                all_regrets[iset] = all_regrets[iset] + regrets
    
    # Compute expected value at this node
    node_value = np.dot(strategy, action_values)
    
    # If this is the updating player's node, compute regrets
    if current_player == player:
        # Counterfactual reach probability (probability opponent and chance reach this node)
        opponent = 1 - player
        cfr_reach = reach_probs[opponent] * reach_probs[2]
        
        # Compute regrets for each action
        regrets = (action_values - node_value) * cfr_reach
        all_regrets[infoset_key] = regrets
    
    return node_value, all_regrets


def cfr(
    env: KuhnPoker,
    num_iters: int = 1000,
    verbose: bool = False
) -> List[StrategyProfile]:
    """Run the CFR algorithm for a given number of iterations.
    
    Args:
        env: The game environment
        num_iters: Number of iterations to run
        verbose: If True, print iteration details
    
    Returns:
        List of average strategy profiles over iterations
    """
    num_players = 2
    
    # Build game tree once
    initial_state = env.init(0)
    root = traverse_tree(env, initial_state)
    
    # Discover all information sets
    def collect_infosets(node, infosets_dict):
        if not node.is_terminal and not node.is_chance and node.infoset_key:
            p = node.player
            if p not in infosets_dict:
                infosets_dict[p] = {}
            if node.infoset_key not in infosets_dict[p]:
                infosets_dict[p][node.infoset_key] = node.legal_actions
        for child in node.children.values():
            collect_infosets(child, infosets_dict)
    
    all_infosets = {}
    collect_infosets(root, all_infosets)
    
    # Initialize cumulative regrets and strategy sums
    cumulative_regrets: List[Dict[InfosetKey, np.ndarray]] = [{}, {}]
    cumulative_strategy: List[Dict[InfosetKey, np.ndarray]] = [{}, {}]
    
    for player in range(num_players):
        for iset_key, actions in all_infosets.get(player, {}).items():
            n_actions = len(actions)
            cumulative_regrets[player][iset_key] = np.zeros(n_actions, dtype=np.float64)
            cumulative_strategy[player][iset_key] = np.zeros(n_actions, dtype=np.float64)
    
    average_strategies_sequence: List[StrategyProfile] = []
    
    for t in range(1, num_iters + 1):
        # Compute current strategies from cumulative regrets using regret matching
        current_strategies = {}
        for player in range(num_players):
            current_strategies[player] = {}
            for infoset_key, regrets in cumulative_regrets[player].items():
                strategy = _regret_matching(regrets)
                current_strategies[player][infoset_key] = strategy
        
        # Update each player
        for player in range(num_players):
            # Compute counterfactual regrets
            reach_probs = [1.0, 1.0, 1.0]  # [prob_p0, prob_p1, prob_chance]
            _, regrets_dict = _compute_counterfactual_regrets(
                root, player, current_strategies, reach_probs
            )
            
            # Update cumulative regrets
            for infoset_key, regrets in regrets_dict.items():
                cumulative_regrets[player][infoset_key] += regrets
            
            # Update cumulative strategy (weighted by iteration)
            for infoset_key, strategy in current_strategies[player].items():
                cumulative_strategy[player][infoset_key] += strategy
        
        # Compute average strategies
        average_strategies = {}
        for player in range(num_players):
            average_strategies[player] = {}
            for infoset_key, strategy_sum in cumulative_strategy[player].items():
                total = np.sum(strategy_sum)
                if total > 0:
                    avg_strategy = strategy_sum / total
                else:
                    # Uniform if never played
                    avg_strategy = np.ones(len(strategy_sum)) / len(strategy_sum)
                average_strategies[player][infoset_key] = avg_strategy
        
        average_strategies_sequence.append(average_strategies)
        
        # Print iteration details if verbose
        if verbose:
            # Compute utility
            combined_strategy = {**average_strategies[0], **average_strategies[1]}
            utilities = evaluate(root, combined_strategy)
            
            print(f"Iter {t}: Utility of avg. strategies: {utilities[0]:.5f}, {utilities[1]:.5f}")
            
            # Print average strategies
            for player in [0, 1]:
                player_name = f"P{player + 1}"
                sorted_infosets = sorted(average_strategies[player].keys(), key=lambda x: str(x))
                
                for infoset_key in sorted_infosets:
                    infoset_str = _format_infoset_for_print(infoset_key)
                    probs = average_strategies[player][infoset_key]
                    
                    # Determine action names
                    player_idx, obs = infoset_key
                    if len(obs) == 7:
                        my_chips = obs[3:5]
                        opp_chips = obs[5:7]
                        
                        if player_idx == 0 and my_chips[0] and opp_chips[1]:
                            action_names = ['Call', 'Fold']
                        elif player_idx == 1 and my_chips[0] and opp_chips[1]:
                            action_names = ['Call', 'Fold']
                        else:
                            action_names = ['Bet', 'Check']
                    else:
                        action_names = [f"Action{i}" for i in range(len(probs))]
                    
                    prob_strs = [f"{action_names[i]}: {probs[i]:.5f}" for i in range(len(probs))]
                    print(f"Iter {t}: Avg. strategy of {player_name} at {infoset_str}: {', '.join(prob_strs)}")
            
            # Print cumulative regrets
            for player in [0, 1]:
                player_name = f"P{player + 1}"
                sorted_infosets = sorted(cumulative_regrets[player].keys(), key=lambda x: str(x))
                
                for infoset_key in sorted_infosets:
                    infoset_str = _format_infoset_for_print(infoset_key)
                    regrets = cumulative_regrets[player][infoset_key]
                    
                    # Determine action names
                    player_idx, obs = infoset_key
                    if len(obs) == 7:
                        my_chips = obs[3:5]
                        opp_chips = obs[5:7]
                        
                        if player_idx == 0 and my_chips[0] and opp_chips[1]:
                            action_names = ['Call', 'Fold']
                        elif player_idx == 1 and my_chips[0] and opp_chips[1]:
                            action_names = ['Call', 'Fold']
                        else:
                            action_names = ['Bet', 'Check']
                    else:
                        action_names = [f"Action{i}" for i in range(len(regrets))]
                    
                    regret_strs = [f"{action_names[i]}: {regrets[i]:.5f}" for i in range(len(regrets))]
                    print(f"Iter {t}: Cumulative regrets of {player_name} at {infoset_str}: {', '.join(regret_strs)}")
            
            print()
    
    return average_strategies_sequence


def cfr_plus(
    env: KuhnPoker,
    num_iters: int = 1000,
    verbose: bool = False
) -> List[StrategyProfile]:
    """Run the CFR+ algorithm for a given number of iterations.
    
    CFR+ differs from standard CFR by:
    1. Flooring negative regrets at 0 after each update
    2. Alternating player updates instead of simultaneous updates
    
    Args:
        env: The game environment
        num_iters: Number of iterations to run
        verbose: If True, print iteration details
    
    Returns:
        List of average strategy profiles over iterations
    """
    num_players = 2
    
    # Build game tree once
    initial_state = env.init(0)
    root = traverse_tree(env, initial_state)
    
    # Discover all information sets
    all_infosets = collect_infosets(root)
    
    # Initialize cumulative regrets and strategy sums
    cumulative_regrets: List[Dict[InfosetKey, np.ndarray]] = [{}, {}]
    cumulative_strategy: List[Dict[InfosetKey, np.ndarray]] = [{}, {}]
    
    for player in range(num_players):
        for iset_key, actions in all_infosets.get(player, {}).items():
            n_actions = len(actions)
            cumulative_regrets[player][iset_key] = np.zeros(n_actions, dtype=np.float64)
            cumulative_strategy[player][iset_key] = np.zeros(n_actions, dtype=np.float64)
    
    average_strategies_sequence: List[StrategyProfile] = []
    
    for t in range(1, num_iters + 1):
        # Alternate between players (CFR+ characteristic)
        for player in range(num_players):
            # Compute current strategies from cumulative regrets using regret matching
            current_strategies = {}
            for p in range(num_players):
                current_strategies[p] = {}
                for infoset_key, regrets in cumulative_regrets[p].items():
                    strategy = _regret_matching(regrets)
                    current_strategies[p][infoset_key] = strategy
            
            # Compute counterfactual regrets for current player
            reach_probs = [1.0, 1.0, 1.0]  # [prob_p0, prob_p1, prob_chance]
            _, regrets_dict = _compute_counterfactual_regrets(
                root, player, current_strategies, reach_probs
            )
            
            # Update cumulative regrets with floor at 0 (CFR+ characteristic)
            for infoset_key, regrets in regrets_dict.items():
                cumulative_regrets[player][infoset_key] += regrets
                # Floor negative regrets at 0
                cumulative_regrets[player][infoset_key] = np.maximum(
                    cumulative_regrets[player][infoset_key], 0
                )
            
            # Update cumulative strategy (weighted by iteration)
            for infoset_key, strategy in current_strategies[player].items():
                cumulative_strategy[player][infoset_key] += strategy
            
            # Compute average strategies
            average_strategies = {}
            for p in range(num_players):
                average_strategies[p] = {}
                for infoset_key, strategy_sum in cumulative_strategy[p].items():
                    total = np.sum(strategy_sum)
                    if total > 0:
                        avg_strategy = strategy_sum / total
                    else:
                        # Uniform if never played
                        avg_strategy = np.ones(len(strategy_sum)) / len(strategy_sum)
                    average_strategies[p][infoset_key] = avg_strategy
            
            # Print iteration details if verbose
            if verbose:
                # Compute utility
                combined_strategy = {**average_strategies[0], **average_strategies[1]}
                utilities = evaluate(root, combined_strategy)
                
                player_name = f"P{player + 1}"
                print(f"Iter {t}, Update of {player_name}: Utility of avg. strategies: {utilities[0]:.5f}, {utilities[1]:.5f}")
                
                # Print average strategies
                for p in [0, 1]:
                    p_name = f"P{p + 1}"
                    sorted_infosets = sorted(average_strategies[p].keys(), key=lambda x: str(x))
                    
                    for infoset_key in sorted_infosets:
                        infoset_str = _format_infoset_for_print(infoset_key)
                        probs = average_strategies[p][infoset_key]
                        
                        # Determine action names
                        player_idx, obs = infoset_key
                        if len(obs) == 7:
                            my_chips = obs[3:5]
                            opp_chips = obs[5:7]
                            
                            if player_idx == 0 and my_chips[0] and opp_chips[1]:
                                action_names = ['Call', 'Fold']
                            elif player_idx == 1 and my_chips[0] and opp_chips[1]:
                                action_names = ['Call', 'Fold']
                            else:
                                action_names = ['Bet', 'Check']
                        else:
                            action_names = [f"Action{i}" for i in range(len(probs))]
                        
                        prob_strs = [f"{action_names[i]}: {probs[i]:.5f}" for i in range(len(probs))]
                        print(f"Iter {t}, Update of {player_name}: Avg. strategy of {p_name} at {infoset_str}: {', '.join(prob_strs)}")
                
                # Print cumulative regrets
                for p in [0, 1]:
                    p_name = f"P{p + 1}"
                    sorted_infosets = sorted(cumulative_regrets[p].keys(), key=lambda x: str(x))
                    
                    for infoset_key in sorted_infosets:
                        infoset_str = _format_infoset_for_print(infoset_key)
                        regrets = cumulative_regrets[p][infoset_key]
                        
                        # Determine action names
                        player_idx, obs = infoset_key
                        if len(obs) == 7:
                            my_chips = obs[3:5]
                            opp_chips = obs[5:7]
                            
                            if player_idx == 0 and my_chips[0] and opp_chips[1]:
                                action_names = ['Call', 'Fold']
                            elif player_idx == 1 and my_chips[0] and opp_chips[1]:
                                action_names = ['Call', 'Fold']
                            else:
                                action_names = ['Bet', 'Check']
                        else:
                            action_names = [f"Action{i}" for i in range(len(regrets))]
                        
                        regret_strs = [f"{action_names[i]}: {regrets[i]:.5f}" for i in range(len(regrets))]
                        print(f"Iter {t}, Update of {player_name}: Cumulative regrets of {p_name} at {infoset_str}: {', '.join(regret_strs)}")
        
        # Store average strategies after both players have updated
        average_strategies = {}
        for p in range(num_players):
            average_strategies[p] = {}
            for infoset_key, strategy_sum in cumulative_strategy[p].items():
                total = np.sum(strategy_sum)
                if total > 0:
                    avg_strategy = strategy_sum / total
                else:
                    avg_strategy = np.ones(len(strategy_sum)) / len(strategy_sum)
                average_strategies[p][infoset_key] = avg_strategy
        
        average_strategies_sequence.append(average_strategies)
        
        if verbose:
            print()
    
    return average_strategies_sequence


def main() -> None:
    """Main function to demonstrate CFR and CFR+."""
    env = KuhnPoker()
    
    print("Running CFR for 10 iterations:")
    print("=" * 80)
    cfr_strategies = cfr(env, num_iters=10, verbose=True)
    
    print("\n" + "=" * 80)
    print("Running CFR+ for 5 iterations:")
    print("=" * 80)
    cfr_plus_strategies = cfr_plus(env, num_iters=5, verbose=True)


if __name__ == '__main__':
    main()
