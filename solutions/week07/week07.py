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
    if memo is None:
        memo = {}
    
    # Create state key for memoization
    state_key = _state_to_key(state)
    if state_key in memo:
        return memo[state_key]
    
    # Create node for this state
    node = GameNode(state, parent, action_taken)
    memo[state_key] = node
    
    # If terminal, return
    if node.is_terminal:
        return node
    
    # Recursively build children
    for action in node.legal_actions:
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
    
    This function assumes weights are already reach-probability-weighted if needed.
    For proper averaging in extensive-form games, weights should be reach probabilities.
    
    Args:
        strategy1: First strategy (dict mapping infoset_key -> action probabilities)
        strategy2: Second strategy (dict mapping infoset_key -> action probabilities)
        weight1: Weight for first strategy (should be reach probability)
        weight2: Weight for second strategy (should be reach probability)
    
    Returns:
        Dict mapping infoset_key -> weighted average action probabilities
    """
    average_strategy: Strategy = {}
    
    # Combine all information sets from both strategies
    all_infosets = set(strategy1.keys()) | set(strategy2.keys())
    
    for infoset_key in all_infosets:
        probs1 = strategy1.get(infoset_key, None)
        probs2 = strategy2.get(infoset_key, None)
        
        if probs1 is not None and probs2 is not None:
            # Both strategies have this infoset - compute weighted average
            # The weights should already account for reach probabilities
            avg_probs = weight1 * probs1 + weight2 * probs2
            total_weight = weight1 + weight2
            if total_weight > 0:
                avg_probs = avg_probs / total_weight
            # Normalize to ensure it's a valid probability distribution
            prob_sum = np.sum(avg_probs)
            if prob_sum > 0:
                avg_probs = avg_probs / prob_sum
        elif probs1 is not None:
            # Only strategy1 has this infoset - use it directly (already normalized)
            avg_probs = probs1.copy()
        else:
            # Only strategy2 has this infoset - use it directly (already normalized)
            avg_probs = probs2.copy()
        
        average_strategy[infoset_key] = avg_probs
    
    return average_strategy


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


def fictitious_play(
    env: KuhnPoker, 
    num_iters: int = 1000
) -> List[StrategyProfile]:
    """Implement Extensive-form Fictitious Play.
    
    Args:
        env: The game environment
        num_iters: Number of iterations
    
    Returns:
        List of average strategy profiles
    """
    num_players: int = 2
    
    # Initialize cumulative strategies weighted by reach probabilities
    cumulative_strategies: List[Strategy] = [{} for _ in range(num_players)]
    cumulative_reach_probs: List[Dict[InfosetKey, float]] = [{} for _ in range(num_players)]
    average_strategies_sequence: List[StrategyProfile] = []
    
    for t in range(1, num_iters + 1):
        # Build game tree
        initial_state = env.init(0)
        root = traverse_tree(env, initial_state)
        
        # For each player, compute best response against average opponent strategy
        current_strategies = {}
        
        for player in range(num_players):
            opponent = 1 - player
            
            # Compute average opponent strategy
            if t == 1:
                # First iteration - use uniform strategy
                opponent_avg_strategy = {}
            else:
                # Average strategy from cumulative (weighted by reach probabilities)
                opponent_avg_strategy = {}
                for infoset_key in cumulative_strategies[opponent].keys():
                    cumulative_probs = cumulative_strategies[opponent][infoset_key]
                    total_reach = cumulative_reach_probs[opponent].get(infoset_key, 1.0)
                    if total_reach > 0:
                        opponent_avg_strategy[infoset_key] = cumulative_probs / total_reach
                    else:
                        # Fallback to uniform if no reach probability
                        num_actions = len(cumulative_probs)
                        opponent_avg_strategy[infoset_key] = np.ones(num_actions) / num_actions
            
            # Compute best response
            best_response = compute_best_response(root, player, opponent_avg_strategy)
            current_strategies[player] = best_response
            
            # Compute reach probabilities for this player's strategy
            reach_probs = compute_reach_probabilities(root, player, best_response)
            
            # Accumulate strategy weighted by reach probabilities
            for infoset_key, action_probs in best_response.items():
                reach_prob = reach_probs.get(infoset_key, 1.0)
                
                if infoset_key not in cumulative_strategies[player]:
                    cumulative_strategies[player][infoset_key] = np.zeros_like(action_probs)
                    cumulative_reach_probs[player][infoset_key] = 0.0
                
                # Weight the strategy by its reach probability
                cumulative_strategies[player][infoset_key] += action_probs * reach_prob
                cumulative_reach_probs[player][infoset_key] += reach_prob
        
        # Compute average strategies (normalized by cumulative reach probabilities)
        average_strategies = {}
        for player in range(num_players):
            average_strategies[player] = {}
            for infoset_key in cumulative_strategies[player].keys():
                cumulative_probs = cumulative_strategies[player][infoset_key]
                total_reach = cumulative_reach_probs[player].get(infoset_key, 1.0)
                if total_reach > 0:
                    avg_probs = cumulative_probs / total_reach
                    # Normalize to ensure valid probability distribution
                    prob_sum = np.sum(avg_probs)
                    if prob_sum > 0:
                        avg_probs = avg_probs / prob_sum
                    average_strategies[player][infoset_key] = avg_probs
                else:
                    # Fallback to uniform
                    num_actions = len(cumulative_probs)
                    average_strategies[player][infoset_key] = np.ones(num_actions) / num_actions
        
        average_strategies_sequence.append(average_strategies)
    
    return average_strategies_sequence

def main() -> None:
    # The implementation of the game is a part of a JAX library called `pgx`.
    # You can find more information about it here: https://www.sotets.uk/pgx/kuhn_poker/
    # We wrap the original implementation to add an explicit chance player and convert
    # everything from JAX arrays to Numpy arrays. There's also a JAX version which you
    # can import using `from kuhn_poker import KuhnPoker` if interested ;)
    env = KuhnPoker()

    # Initialize the environment with a random seed
    state = env.init(0)

    while not (state.terminated or state.truncated):
        if state.is_chance_node:
            uniform_strategy = state.legal_action_mask / np.sum(state.legal_action_mask)
            assert np.allclose(state.chance_strategy, uniform_strategy), (
                'The chance strategy is not uniform!'
            )

        print(state.legal_action_mask)
        # Pick the first legal action
        action = np.argmax(state.legal_action_mask)

        # Take a step in the environment
        state = env.step(state, action)

    assert np.sum(state.rewards) == 0, 'The game is not zero-sum!'
    assert state.terminated or state.truncated, 'The game is not over!'

if __name__ == '__main__':
    main()
