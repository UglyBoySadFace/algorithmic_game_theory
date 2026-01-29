#!/usr/bin/env python3

"""
Week 8: Normal-form and Sequence-form conversion, Sequence-form Linear Programming

This module implements:
1. Conversion of extensive-form games to normal-form representation
2. Conversion of extensive-form games to sequence-form representation
3. Finding Nash equilibria using sequence-form linear programming
4. Converting realization plans to behavioral strategies
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from scipy.optimize import linprog

from solutions.kuhn_poker import KuhnPokerNumpy as KuhnPoker
from solutions.kuhn_poker import State as KuhnPokerState
from solutions.week07.week07 import GameNode, traverse_tree, InfosetKey


def _collect_pure_strategies(
    root_node: GameNode,
    player: int,
    current_strategy: Optional[Dict[InfosetKey, int]] = None
) -> List[Dict[InfosetKey, int]]:
    """Collect all pure strategies for a player.
    
    A pure strategy specifies one action at each information set.
    
    Args:
        root_node: Root of the game tree
        player: Player index
        current_strategy: Current partial strategy being built
    
    Returns:
        List of pure strategies, where each strategy is a dict mapping infoset_key -> action_index
    """
    if current_strategy is None:
        current_strategy = {}
    
    # Collect all information sets for this player
    infosets: Dict[InfosetKey, List[int]] = {}
    
    def collect_infosets(node: GameNode) -> None:
        """Recursively collect all information sets and their legal actions."""
        if node.is_terminal or node.is_chance:
            for child in node.children.values():
                collect_infosets(child)
            return
        
        if node.player == player and node.infoset_key is not None:
            if node.infoset_key not in infosets:
                infosets[node.infoset_key] = list(node.legal_actions)
        
        for child in node.children.values():
            collect_infosets(child)
    
    collect_infosets(root_node)
    
    # Generate all combinations of actions at each information set
    infoset_list = list(infosets.keys())
    
    if len(infoset_list) == 0:
        return [{}]
    
    def generate_strategies(idx: int, current: Dict[InfosetKey, int]) -> List[Dict[InfosetKey, int]]:
        """Recursively generate all pure strategies."""
        if idx == len(infoset_list):
            return [current.copy()]
        
        infoset = infoset_list[idx]
        actions = infosets[infoset]
        
        strategies = []
        for action in actions:
            current[infoset] = action
            strategies.extend(generate_strategies(idx + 1, current))
        
        return strategies
    
    return generate_strategies(0, {})


def _compute_expected_utility(
    root_node: GameNode,
    player: int,
    pure_strategy_profile: Tuple[Dict[InfosetKey, int], Dict[InfosetKey, int]]
) -> float:
    """Compute expected utility for a player given a pure strategy profile.
    
    Args:
        root_node: Root of the game tree
        player: Player index to compute utility for
        pure_strategy_profile: Tuple of pure strategies for both players
    
    Returns:
        Expected utility for the player
    """
    def evaluate_node(node: GameNode) -> float:
        """Recursively evaluate expected utility."""
        if node.is_terminal:
            return node.rewards[player]
        
        if node.is_chance:
            # Chance node - average over chance outcomes
            expected = 0.0
            for action in node.legal_actions:
                prob = node.state.chance_strategy[action]
                expected += prob * evaluate_node(node.children[action])
            return expected
        
        # Player node - follow pure strategy
        current_player = node.player
        strategy = pure_strategy_profile[current_player]
        
        if node.infoset_key in strategy:
            action = strategy[node.infoset_key]
            if action in node.children:
                return evaluate_node(node.children[action])
        
        # If action not found, return 0 (shouldn't happen with valid strategies)
        return 0.0
    
    return evaluate_node(root_node)


def convert_to_normal_form(env: KuhnPoker, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Convert an extensive-form game into an equivalent normal-form representation.

    Args:
        env: The game environment
        seed: Random seed for initializing the game
    
    Returns:
        tuple[np.ndarray, np.ndarray]: A pair of payoff matrices for the two players.
            payoff_matrix_p0[i, j] = utility for player 0 when P0 plays strategy i and P1 plays strategy j
            payoff_matrix_p1[i, j] = utility for player 1 when P0 plays strategy i and P1 plays strategy j
    """
    # Build the game tree
    initial_state = env.init(seed)
    root = traverse_tree(env, initial_state)
    
    # Collect pure strategies for both players
    strategies_p0 = _collect_pure_strategies(root, 0)
    strategies_p1 = _collect_pure_strategies(root, 1)
    
    # Initialize payoff matrices
    num_strategies_p0 = len(strategies_p0)
    num_strategies_p1 = len(strategies_p1)
    
    payoff_matrix_p0 = np.zeros((num_strategies_p0, num_strategies_p1))
    payoff_matrix_p1 = np.zeros((num_strategies_p0, num_strategies_p1))
    
    # Compute payoffs for each pure strategy profile
    for i, strategy_p0 in enumerate(strategies_p0):
        for j, strategy_p1 in enumerate(strategies_p1):
            strategy_profile = (strategy_p0, strategy_p1)
            
            payoff_matrix_p0[i, j] = _compute_expected_utility(root, 0, strategy_profile)
            payoff_matrix_p1[i, j] = _compute_expected_utility(root, 1, strategy_profile)
    
    return payoff_matrix_p0, payoff_matrix_p1


def _collect_sequences(
    root_node: GameNode,
    player: int
) -> Tuple[List[Tuple], Dict[Tuple, int]]:
    """Collect all sequences for a player.
    
    A sequence is a path through the player's information sets.
    The empty sequence is always included at index 0.
    
    Args:
        root_node: Root of the game tree
        player: Player index
    
    Returns:
        Tuple of (sequence_list, sequence_to_index_map)
        - sequence_list: List of sequences where each sequence is tuple of (infoset_key, action) pairs
        - sequence_to_index_map: Dict mapping sequence -> index in the list
    """
    sequences = [()]  # Empty sequence at index 0
    sequence_to_idx = {(): 0}
    
    def traverse_for_sequences(node: GameNode, current_seq: Tuple = ()) -> None:
        """Recursively traverse tree to collect sequences."""
        if node.is_terminal:
            return
        
        if node.is_chance:
            # Skip chance nodes
            for child in node.children.values():
                traverse_for_sequences(child, current_seq)
            return
        
        if node.player == player and node.infoset_key is not None:
            # Extend sequence for each action
            for action in node.legal_actions:
                new_seq = current_seq + ((node.infoset_key, action),)
                if new_seq not in sequence_to_idx:
                    sequence_to_idx[new_seq] = len(sequences)
                    sequences.append(new_seq)
                
                # Continue traversal
                if action in node.children:
                    traverse_for_sequences(node.children[action], new_seq)
        else:
            # Other player's node - continue without extending sequence
            for child in node.children.values():
                traverse_for_sequences(child, current_seq)
    
    traverse_for_sequences(root_node)
    
    return sequences, sequence_to_idx


def _compute_sequence_form_payoff_matrix(
    root_node: GameNode,
    player: int,
    sequences_p0: List[Tuple],
    sequences_p1: List[Tuple],
    seq_to_idx_p0: Dict[Tuple, int],
    seq_to_idx_p1: Dict[Tuple, int]
) -> np.ndarray:
    """Compute the sequence-form payoff matrix for a player.
    
    Args:
        root_node: Root of the game tree
        player: Player index to compute payoffs for
        sequences_p0: List of sequences for player 0
        sequences_p1: List of sequences for player 1
        seq_to_idx_p0: Mapping from sequence to index for player 0
        seq_to_idx_p1: Mapping from sequence to index for player 1
    
    Returns:
        Payoff matrix F where F[i, j] = payoff for player when P0 plays sequence i and P1 plays sequence j
    """
    num_seq_p0 = len(sequences_p0)
    num_seq_p1 = len(sequences_p1)
    
    payoff_matrix = np.zeros((num_seq_p0, num_seq_p1))
    
    def evaluate_node(
        node: GameNode,
        current_seq_p0: Tuple = (),
        current_seq_p1: Tuple = (),
        prob: float = 1.0
    ) -> None:
        """Recursively compute payoffs for sequence pairs."""
        if node.is_terminal:
            # Get indices for current sequences
            idx_p0 = seq_to_idx_p0.get(current_seq_p0, 0)
            idx_p1 = seq_to_idx_p1.get(current_seq_p1, 0)
            
            # Add weighted terminal payoff
            payoff_matrix[idx_p0, idx_p1] += prob * node.rewards[player]
            return
        
        if node.is_chance:
            # Chance node - weight by chance probability
            for action in node.legal_actions:
                chance_prob = node.state.chance_strategy[action]
                child = node.children[action]
                evaluate_node(child, current_seq_p0, current_seq_p1, prob * chance_prob)
            return
        
        # Player node
        current_player = node.player
        infoset = node.infoset_key
        
        for action in node.legal_actions:
            child = node.children[action]
            
            if current_player == 0:
                new_seq = current_seq_p0 + ((infoset, action),) if infoset is not None else current_seq_p0
                evaluate_node(child, new_seq, current_seq_p1, prob)
            else:
                new_seq = current_seq_p1 + ((infoset, action),) if infoset is not None else current_seq_p1
                evaluate_node(child, current_seq_p0, new_seq, prob)
    
    # Start recursion from root with empty sequences
    evaluate_node(root_node)
    
    return payoff_matrix


def _compute_realization_plan_constraints(
    sequences: List[Tuple],
    seq_to_idx: Dict[Tuple, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute realization-plan constraints for a player.
    
    The constraints are: E @ r = e
    where r is the realization plan (probability of reaching each sequence)
    
    For each information set h, we have the constraint:
    r(h) = sum over actions a at h of r(h, a)
    where r(h) is the reach probability to information set h (parent sequence)
    and r(h, a) is the reach probability to sequence ending with (h, a)
    
    Args:
        sequences: List of sequences for the player
        seq_to_idx: Mapping from sequence to index
    
    Returns:
        Tuple of (E, e) where E @ r = e are the realization plan constraints
    """
    num_sequences = len(sequences)
    
    # Group sequences by their parent
    children_by_parent: Dict[Tuple, List[Tuple]] = defaultdict(list)
    
    for seq in sequences:
        if len(seq) == 0:
            # Empty sequence has no parent
            continue
        elif len(seq) == 1:
            # Direct child of empty sequence
            parent = ()
        else:
            # Parent is all but last element
            parent = seq[:-1]
        
        children_by_parent[parent].append(seq)
    
    # Create constraints
    constraints_list = []
    rhs_list = []
    
    # First constraint: empty sequence has probability 1
    constraint = np.zeros(num_sequences)
    constraint[0] = 1
    constraints_list.append(constraint)
    rhs_list.append(1)
    
    # r(parent) = sum_children r(child)
    for parent_seq in sequences:
        children = children_by_parent.get(parent_seq, [])
        if len(children) > 0:
            # Constraint: r(parent) - sum(r(children)) = 0
            constraint = np.zeros(num_sequences)
            parent_idx = seq_to_idx[parent_seq]
            constraint[parent_idx] = 1
            
            for child_seq in children:
                child_idx = seq_to_idx[child_seq]
                constraint[child_idx] = -1
            
            constraints_list.append(constraint)
            rhs_list.append(0)
    
    E = np.array(constraints_list)
    e = np.array(rhs_list)
    
    return E, e


def convert_to_sequence_form(
    env: KuhnPoker,
    seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert an extensive-form game into its sequence-form representation.

    The sequence-form representation consists of:
        - The sequence-form payoff matrices for both players
        - The realization-plan constraint matrices and vectors for both players

    Args:
        env: The game environment
        seed: Random seed for initializing the game

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            (F_p0, F_p1, E_p0, e_p0, E_p1, e_p1)
            - F_p0: Sequence-form payoff matrix for player 0
            - F_p1: Sequence-form payoff matrix for player 1
            - E_p0, e_p0: Realization-plan constraints for player 0 (E_p0 @ r_p0 = e_p0)
            - E_p1, e_p1: Realization-plan constraints for player 1 (E_p1 @ r_p1 = e_p1)
    """
    # Build the game tree
    initial_state = env.init(seed)
    root = traverse_tree(env, initial_state)
    
    # Collect sequences for both players
    sequences_p0, seq_to_idx_p0 = _collect_sequences(root, 0)
    sequences_p1, seq_to_idx_p1 = _collect_sequences(root, 1)
    
    # Compute sequence-form payoff matrices
    F_p0 = _compute_sequence_form_payoff_matrix(
        root, 0, sequences_p0, sequences_p1, seq_to_idx_p0, seq_to_idx_p1
    )
    F_p1 = _compute_sequence_form_payoff_matrix(
        root, 1, sequences_p0, sequences_p1, seq_to_idx_p0, seq_to_idx_p1
    )
    
    # Compute realization-plan constraints
    E_p0, e_p0 = _compute_realization_plan_constraints(root, 0, sequences_p0, seq_to_idx_p0)
    E_p1, e_p1 = _compute_realization_plan_constraints(root, 1, sequences_p1, seq_to_idx_p1)
    
    return F_p0, F_p1, E_p0, e_p0, E_p1, e_p1


def find_nash_equilibrium_sequence_form(
    env: KuhnPoker,
    seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Find a Nash equilibrium in a zero-sum extensive-form game using Sequence-form LP.

    This function converts the game to sequence-form and solves the resulting
    linear program to find a Nash equilibrium.

    Args:
        env: The game environment
        seed: Random seed for initializing the game

    Returns:
        tuple[np.ndarray, np.ndarray]:
            A pair of realization plans for the two players representing a Nash equilibrium.
    """
    # Convert to sequence form
    F_p0, F_p1, E_p0, e_p0, E_p1, e_p1 = convert_to_sequence_form(env, seed)
    
    # For zero-sum games: F_p1 = -F_p0
    # We solve for player 0's optimal strategy against player 1
    
    # Player 0 maximizes: min_j (F_p0 @ r_p0)_j subject to E_p0 @ r_p0 = e_p0, r_p0 >= 0
    # This is equivalent to:
    # max v such that F_p0 @ r_p0 >= v * 1, E_p0 @ r_p0 = e_p0, r_p0 >= 0
    
    # Convert to minimization for scipy.linprog:
    # min -v such that -F_p0 @ r_p0 + v * 1 <= 0, E_p0 @ r_p0 = e_p0, r_p0 >= 0
    
    num_seq_p0 = F_p0.shape[0]
    num_seq_p1 = F_p0.shape[1]
    
    # Variables: [r_p0 (size num_seq_p0), v (size 1)]
    # Objective: minimize -v
    c = np.zeros(num_seq_p0 + 1)
    c[-1] = -1  # Maximize v
    
    # Inequality constraints: -F_p0^T @ r_p0 + v * 1 <= 0
    # Or: F_p0^T @ r_p0 - v * 1 >= 0
    # For linprog (A_ub @ x <= b_ub): -F_p0^T @ r_p0 + v * 1 <= 0
    A_ub = np.zeros((num_seq_p1, num_seq_p0 + 1))
    A_ub[:, :num_seq_p0] = -F_p0.T
    A_ub[:, -1] = 1
    b_ub = np.zeros(num_seq_p1)
    
    # Equality constraints: E_p0 @ r_p0 = e_p0
    A_eq = np.zeros((E_p0.shape[0], num_seq_p0 + 1))
    A_eq[:, :num_seq_p0] = E_p0
    b_eq = e_p0
    
    # Bounds: r_p0 >= 0, v unbounded
    bounds = [(0, None)] * num_seq_p0 + [(None, None)]
    
    # Solve LP for player 0
    result_p0 = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if not result_p0.success:
        raise ValueError("Failed to solve LP for player 0")
    
    r_p0 = result_p0.x[:num_seq_p0]
    
    # Now solve for player 1 (minimizing player)
    # Player 1 minimizes: max_i (F_p1 @ r_p1)_i = max_i (-F_p0 @ r_p1)_i
    # Equivalent to: min u such that -F_p0 @ r_p1 <= u * 1, E_p1 @ r_p1 = e_p1, r_p1 >= 0
    
    # Variables: [r_p1 (size num_seq_p1), u (size 1)]
    # Objective: minimize u
    c = np.zeros(num_seq_p1 + 1)
    c[-1] = 1  # Minimize u
    
    # Inequality constraints: -F_p0 @ r_p1 - u * 1 <= 0
    A_ub = np.zeros((num_seq_p0, num_seq_p1 + 1))
    A_ub[:, :num_seq_p1] = -F_p0
    A_ub[:, -1] = -1
    b_ub = np.zeros(num_seq_p0)
    
    # Equality constraints: E_p1 @ r_p1 = e_p1
    A_eq = np.zeros((E_p1.shape[0], num_seq_p1 + 1))
    A_eq[:, :num_seq_p1] = E_p1
    b_eq = e_p1
    
    # Bounds: r_p1 >= 0, u unbounded
    bounds = [(0, None)] * num_seq_p1 + [(None, None)]
    
    # Solve LP for player 1
    result_p1 = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if not result_p1.success:
        raise ValueError("Failed to solve LP for player 1")
    
    r_p1 = result_p1.x[:num_seq_p1]
    
    return r_p0, r_p1


def convert_realization_plan_to_behavioural_strategy(
    env: KuhnPoker,
    player: int,
    realization_plan: np.ndarray,
    seed: int = 0
) -> Dict[InfosetKey, np.ndarray]:
    """Convert a realization plan to a behavioural strategy.
    
    Args:
        env: The game environment
        player: Player index
        realization_plan: Realization plan (probability of reaching each sequence)
        seed: Random seed for initializing the game
    
    Returns:
        Dict mapping infoset_key -> action probabilities (behavioral strategy)
    """
    # Build the game tree
    initial_state = env.init(seed)
    root = traverse_tree(env, initial_state)
    
    # Collect sequences
    sequences, seq_to_idx = _collect_sequences(root, player)
    
    # Build parent-child relationships
    sequence_children: Dict[Tuple, List[Tuple]] = defaultdict(list)
    
    for seq in sequences:
        if len(seq) == 0:
            continue
        elif len(seq) == 1:
            parent = ()
        else:
            parent = seq[:-1]
        
        sequence_children[parent].append(seq)
    
    # Collect all information sets and group sequences by their terminal information set
    infoset_to_sequences: Dict[InfosetKey, Dict[int, Tuple]] = defaultdict(dict)
    
    for seq in sequences:
        if len(seq) > 0:
            last_infoset, last_action = seq[-1]
            infoset_to_sequences[last_infoset][last_action] = seq
    
    # Convert realization plan to behavioral strategy
    behavioral_strategy: Dict[InfosetKey, np.ndarray] = {}
    
    for infoset, action_to_seq in infoset_to_sequences.items():
        actions = sorted(action_to_seq.keys())
        
        # Find parent sequence (common prefix of all sequences at this infoset)
        seq_list = list(action_to_seq.values())
        if len(seq_list[0]) > 0:
            parent_seq = seq_list[0][:-1]
        else:
            parent_seq = ()
        
        # Get parent probability
        parent_idx = seq_to_idx[parent_seq]
        parent_prob = realization_plan[parent_idx]
        
        # Compute behavioral probabilities
        action_probs = np.zeros(len(actions))
        
        for i, action in enumerate(actions):
            seq = action_to_seq[action]
            seq_idx = seq_to_idx[seq]
            seq_prob = realization_plan[seq_idx]
            
            # Behavioral probability = seq_prob / parent_prob
            if parent_prob > 1e-10:
                action_probs[i] = seq_prob / parent_prob
            else:
                action_probs[i] = 1.0 / len(actions)  # Uniform if unreachable
        
        # Normalize to ensure valid probability distribution
        prob_sum = np.sum(action_probs)
        if prob_sum > 1e-10:
            action_probs = action_probs / prob_sum
        else:
            action_probs = np.ones(len(actions)) / len(actions)
        
        behavioral_strategy[infoset] = action_probs
    
    return behavioral_strategy


def main() -> None:
    return None

if __name__ == '__main__':
    main()
