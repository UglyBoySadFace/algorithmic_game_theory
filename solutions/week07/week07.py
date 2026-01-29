#!/usr/bin/env python3

"""
Extensive-form games assignments.

Starting this week, the templates will no longer contain exact function
signatures and there will not be any automated tests like we had for the
normal-form games assignments. Instead, we will provide sample outputs
produced by the reference implementations which you can use to verify
your solutions. The reason for this change is that there are many valid
ways to represent game trees (e.g. flat array-based vs. pointer-based),
information sets and strategies in extensive-form games. Figuring out
the most suitable representations is an important part of assignments
in this block. Unfortunately, this freedom makes automated testing
pretty much impossible.
"""

import sys
import os
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
from numpy.typing import NDArray
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt

from solutions.kuhn_poker import KuhnPokerNumpy as KuhnPoker
from solutions.kuhn_poker import State as KuhnPokerState

# Type aliases for clarity
InfosetKey = Tuple[int, Tuple[Any, ...]]
Strategy = Dict[InfosetKey, NDArray[np.float64]]
StrategyProfile = Dict[int, Strategy]


class GameNode:
    """Represents a node in the extensive-form game tree."""
    
    def __init__(
        self, 
        state: KuhnPokerState, 
        parent: Optional['GameNode'] = None, 
        action_taken: Optional[int] = None
    ) -> None:
        
        self.state: KuhnPokerState = state
        self.parent: Optional['GameNode'] = parent
        self.action_taken: Optional[int] = action_taken
        self.children: Dict[int, 'GameNode'] = {}
        self.player: int = int(state.current_player)
        self.is_terminal: bool = bool(state.terminated or state.truncated)
        self.is_chance: bool = bool(state.is_chance_node)
        self.legal_actions: NDArray[np.int64] = np.where(state.legal_action_mask)[0]
        self.rewards: NDArray[np.float64] = state.rewards
        self.infoset_key: Optional[InfosetKey] = self._compute_infoset_key()
    
    def _compute_infoset_key(self) -> Optional[InfosetKey]:
        """Compute a key that identifies this node's information set."""
        if self.is_terminal or self.is_chance:
            return None
        
        # Use the observation as the information set identifier
        return (self.player, tuple(self.state.observation))


def traverse_tree(
    env: KuhnPoker, 
    state: KuhnPokerState, 
    parent: Optional[GameNode] = None, 
    action_taken: Optional[int] = None, 
    memo: Optional[Dict[Tuple[Any, ...], GameNode]] = None
) -> GameNode:
    """Build a full extensive-form game tree for a given game.
    
    Args:
        env: The game environment
        state: Current game state
        parent: Parent node (optional)
        action_taken: Action that led to this state (optional)
        memo: Memoization dict to avoid rebuilding identical states
    
    Returns:
        GameNode representing the root of the (sub)tree
    """
    # Create node for this state
    node = GameNode(state, parent, action_taken)
    
    # If terminal, return
    if node.is_terminal:
        return node
    
    # Recursively build children
    for action in reversed(node.legal_actions):
        next_state = env.step(state, int(action))
        child = traverse_tree(env, next_state, node, action, memo)
        node.children[action] = child
    
    return node


def _state_to_key(state: KuhnPokerState) -> Tuple[Tuple[str, Any], ...]:
    """Convert state to a hashable key for memoization."""
    key_parts: List[Tuple[str, Any]] = [
        ('player', int(state.current_player)),
        ('terminated', bool(state.terminated)),
        ('truncated', bool(state.truncated)),
        ('observation', tuple(state.observation)),
        ('legal_actions', tuple(state.legal_action_mask))
    ]
    
    return tuple(key_parts)


def evaluate(root_node: GameNode, strategies: Strategy) -> NDArray[np.float64]:
    """Compute the expected utility of each player in an extensive-form game.
    
    Args:
        root_node: Root of the game tree
        strategies: Dict mapping infoset_key -> action probabilities
    
    Returns:
        np.ndarray of expected utilities for each player
    """
    num_players: int = 2  # Assuming 2-player game
    
    def evaluate_node(node: GameNode, reach_prob: float = 1.0) -> NDArray[np.float64]:
        """Recursively compute expected utilities."""
        if node.is_terminal:
            return node.rewards * reach_prob
        
        expected_utilities: NDArray[np.float64] = np.zeros(num_players, dtype=np.float64)
        
        if node.is_chance:
            # Chance node - use chance strategy
            for action in node.legal_actions:
                prob = node.state.chance_strategy[action]
                child = node.children[action]
                expected_utilities += evaluate_node(child, reach_prob * prob)
        else:
            # Player node - use strategy
            player = node.player
            infoset_key = node.infoset_key
            
            if infoset_key in strategies:
                action_probs = strategies[infoset_key]
            else:
                # Uniform strategy if not specified
                action_probs = np.ones(len(node.legal_actions), dtype=np.float64) / len(node.legal_actions)
            
            for i, action in enumerate(node.legal_actions):
                prob = action_probs[i] if i < len(action_probs) else 0.0
                child = node.children[action]
                expected_utilities += evaluate_node(child, reach_prob * prob)
        
        return expected_utilities
    
    return evaluate_node(root_node)


def compute_best_response(
    root_node: GameNode, 
    player: int, 
    opponent_strategy: Strategy
) -> Strategy:
    """Compute a best response strategy for a given player against a fixed opponent's strategy.
    
    Args:
        root_node: Root of the game tree
        player: Player index (0 or 1)
        opponent_strategy: Dict mapping infoset_key -> action probabilities for opponent
    
    Returns:
        Dict mapping infoset_key -> action probabilities (best response strategy)
    """
    best_response: Strategy = {}
    infoset_values: Dict[InfosetKey, float] = {}
    
    def compute_value(node: GameNode, reach_prob_opponent: float = 1.0) -> float:
        """Compute the value of each node and construct best response."""
        if node.is_terminal:
            return node.rewards[player]
        
        if node.is_chance:
            # Chance node
            expected_value = 0.0
            for action in node.legal_actions:
                prob = node.state.chance_strategy[action]
                child = node.children[action]
                expected_value += prob * compute_value(child, reach_prob_opponent)
            return expected_value
        
        if node.player == player:
            # Player's decision node - maximize
            action_values: List[float] = []
            for action in node.legal_actions:
                child = node.children[action]
                value = compute_value(child, reach_prob_opponent)
                action_values.append(value)

            # Best response: play best action with probability 1
            best_action_idx = np.argmax(action_values)
            max_value = action_values[best_action_idx]
            
            # Store best response strategy for this information set
            infoset_key = node.infoset_key
            if infoset_key is not None:
                action_probs = np.zeros(len(node.legal_actions), dtype=np.float64)
                action_probs[best_action_idx] = 1.0
                best_response[infoset_key] = action_probs
                infoset_values[infoset_key] = max_value
            
            return max_value
        else:
            # Opponent's decision node
            infoset_key = node.infoset_key
            
            if infoset_key in opponent_strategy:
                action_probs = opponent_strategy[infoset_key]
            else:
                # Uniform if not specified
                action_probs = np.ones(len(node.legal_actions), dtype=np.float64) / len(node.legal_actions)
            
            expected_value = 0.0
            for i, action in enumerate(node.legal_actions):
                prob = action_probs[i] if i < len(action_probs) else 0.0
                child = node.children[action]
                expected_value += prob * compute_value(child, reach_prob_opponent * prob)
            
            return expected_value
    
    compute_value(root_node)
    return best_response


def compute_average_strategy(
    strategy1: Strategy,
    strategy2: Strategy,
    weight1: float = 0.5,
    weight2: float = 0.5
) -> Strategy:
    """Compute a weighted average of a pair of behavioural strategies for a given player.
    
    Args:
        strategy1: First strategy (dict mapping infoset_key -> action probabilities)
        strategy2: Second strategy (dict mapping infoset_key -> action probabilities)
        weight1: Weight for first strategy (default 0.5)
        weight2: Weight for second strategy (default 0.5)
    
    Returns:
        Averaged strategy (dict mapping infoset_key -> action probabilities)
    """
    averaged_strategy: Strategy = {}
    
    # Get all information sets from both strategies
    all_infosets = set(strategy1.keys()) | set(strategy2.keys())
    
    for infoset_key in all_infosets:
        if infoset_key in strategy1 and infoset_key in strategy2:
            # Both strategies have this information set
            avg_probs = weight1 * strategy1[infoset_key] + weight2 * strategy2[infoset_key]
        elif infoset_key in strategy1:
            # Only strategy1 has this information set
            avg_probs = weight1 * strategy1[infoset_key]
        else:
            # Only strategy2 has this information set
            avg_probs = weight2 * strategy2[infoset_key]
        
        # Normalize to ensure valid probability distribution
        prob_sum = np.sum(avg_probs)
        if prob_sum > 0:
            avg_probs = avg_probs / prob_sum
        
        averaged_strategy[infoset_key] = avg_probs
    
    return averaged_strategy

def compute_exploitability(
    env: KuhnPoker,
    strategy_profiles: List[StrategyProfile],
    plot: bool = True
) -> NDArray[np.float64]:
    """Compute and plot the exploitability of a sequence of strategy profiles.
    
    Exploitability is defined as the sum over all players of:
    (best_response_value - current_strategy_value)
    
    Args:
        env: The game environment
        strategy_profiles: List of strategy profiles (one per iteration)
        plot: Whether to plot the exploitability over iterations (default True)
    
    Returns:
        Array of exploitability values for each iteration
    """
    exploitabilities: List[float] = []
    
    # Build game tree once
    initial_state = env.init(0)
    root = traverse_tree(env, initial_state)
    
    for strategy_profile in strategy_profiles:
        total_exploitability = 0.0
        
        # For each player, compute exploitability
        for player in [0, 1]:
            opponent = 1 - player
            
            # Get current strategies
            player_strategy = strategy_profile[player]
            opponent_strategy = strategy_profile[opponent]
            
            # Compute value of current strategy profile for this player
            combined_strategy = {**player_strategy, **opponent_strategy}
            current_values = evaluate(root, combined_strategy)
            current_value = current_values[player]
            
            # Compute best response against opponent's strategy
            best_response = compute_best_response(root, player, opponent_strategy)
            
            # Compute value of best response
            combined_br_strategy = {**best_response, **opponent_strategy}
            br_values = evaluate(root, combined_br_strategy)
            br_value = br_values[player]
            
            # Exploitability contribution for this player
            player_exploitability = br_value - current_value
            total_exploitability += player_exploitability
        
        exploitabilities.append(total_exploitability)
    
    exploitabilities_array = np.array(exploitabilities)
    
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(exploitabilities) + 1), exploitabilities_array)
        plt.xlabel('Iteration')
        plt.ylabel('Exploitability')
        plt.title('Exploitability over Fictitious Play Iterations')
        plt.grid(True)
        plt.yscale('log')
        plt.savefig('exploitability_plot.png')
    
    return exploitabilities_array

def compute_reach_probabilities(
    root_node: GameNode,
    player: int,
    strategy: Strategy
) -> Dict[InfosetKey, float]:
    """Compute reach probabilities for each information set of a player.
    
    Args:
        root_node: Root of the game tree
        player: Player index
        strategy: Strategy profile for all players
    
    Returns:
        Dict mapping infoset_key -> reach probability
    """
    reach_probs: Dict[InfosetKey, float] = {}
    
    def traverse(node: GameNode, reach_prob: float) -> None:
        """Recursively compute reach probabilities."""
        if node.is_terminal:
            return
        
        # Track reach probability for this player's information sets
        if not node.is_chance and node.player == player and node.infoset_key is not None:
            if node.infoset_key not in reach_probs:
                reach_probs[node.infoset_key] = 0.0
            reach_probs[node.infoset_key] += reach_prob
        
        # Recurse to children
        if node.is_chance:
            # Chance node - use chance probabilities
            for action in node.legal_actions:
                prob = node.state.chance_strategy[action]
                child = node.children[action]
                traverse(child, reach_prob * prob)
        else:
            # Player node - use strategy
            infoset_key = node.infoset_key
            if infoset_key in strategy:
                action_probs = strategy[infoset_key]
            else:
                # Uniform if not specified
                action_probs = np.ones(len(node.legal_actions), dtype=np.float64) / len(node.legal_actions)
            
            for i, action in enumerate(node.legal_actions):
                prob = action_probs[i] if i < len(action_probs) else 0.0
                child = node.children[action]
                # Only multiply by action probability if this is the target player
                if node.player == player:
                    traverse(child, reach_prob * prob)
                else:
                    traverse(child, reach_prob)
    
    traverse(root_node, 1.0)
    return reach_probs


def _format_infoset_for_print(infoset_key: InfosetKey) -> str:
    """Convert infoset key to readable format like ('J', '') or ('', 'K', 'Bet')."""
    player, obs = infoset_key
    # obs is a 7-element boolean array:
    # Index 0-2: card (J/Q/K)
    # Index 3-4: current player chips (0 or 1)
    # Index 5-6: opponent chips (0 or 1)
    
    obs_array = np.array(obs, dtype=bool)
    
    # Decode card
    card_map = {0: 'J', 1: 'Q', 2: 'K'}
    card = ''
    for i in range(3):
        if obs_array[i]:
            card = card_map[i]
            break
    
    # Decode pot state to determine history
    my_chips = 1 if obs_array[4] else 0
    opp_chips = 1 if obs_array[6] else 0
    
    # Construct history based on pot state and player
    # Player 0 moves first, then player 1
    if player == 0:
        # Player 0's turn
        if my_chips == 0 and opp_chips == 0:
            # First decision - no history
            return f"('{card}', '')"
        elif my_chips == 0 and opp_chips == 1:
            # P1 bet, now P0 responds
            return f"('{card}', '', 'Check', 'Bet')"
        else:
            # Shouldn't reach other states for P0
            return f"('{card}', '')"
    else:
        # Player 1's turn
        if my_chips == 0 and opp_chips == 0:
            # P0 passed, now P1 decides
            return f"('', '{card}', 'Check')"
        elif my_chips == 0 and opp_chips == 1:
            # P0 bet, now P1 responds
            return f"('', '{card}', 'Bet')"
        else:
            # Shouldn't reach other states for P1
            return f"('', '{card}', 'Check')"

def fictitious_play(
    env: KuhnPoker, 
    num_iters: int = 1000,
    verbose: bool = False
) -> List[StrategyProfile]:
    """Implement Extensive-form Fictitious Play.
    
    Args:
        env: The game environment
        num_iters: Number of iterations
        verbose: If True, print iteration details like reference output
    
    Returns:
        List of average strategy profiles
    """
    num_players: int = 2
    
    # Build game tree once to discover all infosets
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
    
    # Initialize cumulative strategies using simple averaging
    # Start with uniform strategies
    cumulative_strategies: List[Strategy] = [{}, {}]
    
    for player in range(num_players):
        for iset_key, actions in all_infosets.get(player, {}).items():
            n_actions = len(actions)
            cumulative_strategies[player][iset_key] = np.ones(n_actions, dtype=np.float64) / n_actions
    
    average_strategies_sequence: List[StrategyProfile] = []
    
    # Store best responses for printing
    best_responses_per_iter: List[Dict[int, Strategy]] = []


    # Build game tree
    root = traverse_tree(env, env.init(0))

    for t in range(1, num_iters + 1):
         # Compute current average strategies from cumulative (simple average over iterations)
        average_strategies = {}
        for player in range(num_players):
            average_strategies[player] = {}
            for infoset_key in cumulative_strategies[player].keys():
                cumulative_probs = cumulative_strategies[player][infoset_key]
                # Simple average: divide by iteration number
                avg_probs = cumulative_probs / t
                # Normalize to ensure valid probability distribution
                prob_sum = np.sum(avg_probs)
                if prob_sum > 0:
                    avg_probs = avg_probs / prob_sum
                average_strategies[player][infoset_key] = avg_probs
        
        # For each player, compute best response against average opponent strategy
        current_strategies = {}
        
        for player in range(num_players):
            opponent = 1 - player
            
            # Use opponent's current average strategy
            opponent_avg_strategy = average_strategies[opponent]
            
            # Compute best response
            best_response = compute_best_response(root, player, opponent_avg_strategy)
            current_strategies[player] = best_response
            
            # Update cumulative strategy (simple sum, will divide by t in next iteration)
            for infoset_key, action_probs in best_response.items():
                if infoset_key not in cumulative_strategies[player]:
                    cumulative_strategies[player][infoset_key] = np.zeros_like(action_probs)
                
                # Simple accumulation (no reach probability weighting)
                cumulative_strategies[player][infoset_key] += action_probs
        
        average_strategies_sequence.append(average_strategies)
        best_responses_per_iter.append(current_strategies)
        
        # Print iteration details if verbose
        if verbose:
            # Compute utility
            combined_strategy = {**average_strategies[0], **average_strategies[1]}
            utilities = evaluate(root, combined_strategy)
            print(f"Iter {t}: Utility of avg. strategies: {utilities[0]:.5f}, {utilities[1]:.5f}")
            
            # Print average strategies
            for player in [0, 1]:
                player_name = f"P{player + 1}"
                # Sort infosets for consistent output
                sorted_infosets = sorted(average_strategies[player].keys(), 
                                        key=lambda x: (x[1]))  # Sort by observation
                
                for infoset_key in sorted_infosets:
                    probs = average_strategies[player][infoset_key]
                    infoset_str = _format_infoset_for_print(infoset_key)
                    
                    # Determine action names based on observation
                    obs = np.array(infoset_key[1], dtype=bool)
                    my_chips = 1 if obs[4] else 0
                    opp_chips = 1 if obs[6] else 0
                    
                    # If opponent has bet (opp_chips=1 and my_chips=0), actions are Call/Fold
                    if player == 0 and my_chips == 0 and opp_chips == 1:
                        action_names_local = ['Call', 'Fold']
                    elif player == 1 and my_chips == 0 and opp_chips == 1:
                        action_names_local = ['Call', 'Fold']
                    else:
                        action_names_local = ['Bet', 'Check']
                    
                    prob_strs = [f"{action_names_local[i]}: {probs[i]:.5f}" 
                                for i in range(len(probs))]
                    print(f"Iter {t}: Avg. strategy of {player_name} at {infoset_str}: {', '.join(prob_strs)}")
            
            # Print best responses
            for player in [0, 1]:
                player_name = f"P{player + 1}"
                opponent_name = f"P{2 - player}"
                sorted_infosets = sorted(current_strategies[player].keys(),
                                        key=lambda x: (x[1]))
                
                for infoset_key in sorted_infosets:
                    probs = current_strategies[player][infoset_key]
                    infoset_str = _format_infoset_for_print(infoset_key)
                    
                    # Determine action names
                    obs = np.array(infoset_key[1], dtype=bool)
                    my_chips = 1 if obs[4] else 0
                    opp_chips = 1 if obs[6] else 0
                    
                    # If opponent has bet (opp_chips=1 and my_chips=0), actions are Call/Fold
                    if player == 0 and my_chips == 0 and opp_chips == 1:
                        action_names_local = ['Call', 'Fold']
                    elif player == 1 and my_chips == 0 and opp_chips == 1:
                        action_names_local = ['Call', 'Fold']
                    else:
                        action_names_local = ['Bet', 'Check']
                    
                    prob_strs = [f"{action_names_local[i]}: {probs[i]:.5f}" 
                                for i in range(len(probs))]
                    print(f"Iter {t}: BR of {player_name} against {opponent_name}'s avg. strategy at {infoset_str}: {', '.join(prob_strs)}")
            
            print()  # Empty line between iterations
    
    return average_strategies_sequence

def main() -> None:
    # The implementation of the game is a part of a JAX library called `pgx`.
    # You can find more information about it here: https://www.sotets.uk/pgx/kuhn_poker/
    # We wrap the original implementation to add an explicit chance player and convert
    # everything from JAX arrays to Numpy arrays. There's also a JAX version which you
    # can import using `from kuhn_poker import KuhnPoker` if interested ;)
    env = KuhnPoker()

    # Test fictitious play with verbose output
    print("Running Fictitious Play with 10 iterations:\n")
    strategies = fictitious_play(env, num_iters=10, verbose=True)
    
    print(f"\nCompleted {len(strategies)} iterations of Fictitious Play.")

    env = KuhnPoker()
    root = traverse_tree(env, env.init(213123))
    # get all infosets
    all_infosets = {}
    def collect_infosets(node, infosets_dict):
        if not node.is_terminal and not node.is_chance and node.infoset_key:
            p = node.player
            if p not in infosets_dict:
                infosets_dict[p] = {}
            if node.infoset_key not in infosets_dict[p]:
                infosets_dict[p][node.infoset_key] = node.legal_actions
        for child in node.children.values():
            collect_infosets(child, infosets_dict)
    
    collect_infosets(root, all_infosets)
    for player in all_infosets:
        print(f"Player {player} infosets:")
        for infoset in all_infosets[player]:
            print(f"  {_format_infoset_for_print(infoset)}")


    env = KuhnPoker()
    exploitabilities = compute_exploitability(env, strategies, plot=True)
    print(f"Exploitabilities over iterations: {exploitabilities}")



if __name__ == '__main__':
    main()
