#!/usr/bin/env python3

import sys
import os
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
from solutions.week01 import week01
from solutions.week04 import week04


def double_oracle(
    row_matrix: np.ndarray, eps: float, rng: np.random.Generator
) -> tuple[list[np.ndarray, np.ndarray], list[np.ndarray, np.ndarray]]:
    """Run Double Oracle until a termination condition is met.

    The reference implementation generates the initial restricted game by
    randomly sampling one pure action for each player using `rng.integers`.

    The algorithm terminates when either:
        1. the difference between the upper and the lower bound on the game value drops below `eps`
        2. both players' best responses are already contained in the current restricted game

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    eps : float
        The required accuracy for the approximate Nash equilibrium
    rng : np.random.Generator
        A random number generator

    Returns
    -------
    tuple[list[np.ndarray, np.ndarray], list[np.ndarray, np.ndarray]]
        A tuple containing a sequence of strategy profiles and a sequence of corresponding supports
    """
    num_rows, num_cols = row_matrix.shape
    
    # Initialize with one random pure strategy for each player
    row_support = [rng.integers(0, num_rows)]
    col_support = [rng.integers(0, num_cols)]
    
    # Lists to store the sequence of strategies and supports
    strategies = []
    supports = []
    
    while True:
        # Create the restricted game
        restricted_row_matrix = row_matrix[np.ix_(row_support, col_support)]
        
        # Solve Nash equilibrium in the restricted game
        restricted_row_strategy, restricted_col_strategy = week04.find_nash_equilibrium(restricted_row_matrix)
        
        # Expand strategies to full game dimension
        row_strategy = np.zeros(num_rows, dtype=np.float64)
        col_strategy = np.zeros(num_cols, dtype=np.float64)
        
        for i, action in enumerate(row_support):
            row_strategy[action] = restricted_row_strategy[i]
        for j, action in enumerate(col_support):
            col_strategy[action] = restricted_col_strategy[j]
        
        # Store current strategy and support
        strategies.append((row_strategy.copy(), col_strategy.copy()))
        supports.append((np.array(row_support, dtype=np.int64), np.array(col_support, dtype=np.int64)))
        
        # Find best responses in the full game
        row_best_response = week01.calculate_best_response_against_col(row_matrix, col_strategy)
        # For zero-sum games, col_matrix = -row_matrix
        col_best_response = week01.calculate_best_response_against_row(-row_matrix, row_strategy)
        
        # Get the pure actions (indices) from best responses
        row_br_action = np.argmax(row_best_response)
        col_br_action = np.argmax(col_best_response)
        
        # Calculate bounds
        # Upper bound: what row can get with best response against col_strategy
        upper_bound = np.dot(row_matrix[row_br_action, :], col_strategy)
        
        # Lower bound: what col can force row to (minimum over col's actions)
        lower_bound = np.min([np.dot(row_strategy, row_matrix[:, j]) for j in range(num_cols)])
        
        # Check termination condition 1: gap between bounds is small
        if upper_bound - lower_bound < eps:
            break
        
        # Check if best responses are already in the support
        row_br_in_support = row_br_action in row_support
        col_br_in_support = col_br_action in col_support
        
        # Check termination condition 2: both best responses already in support
        if row_br_in_support and col_br_in_support:
            break
        
        # Add new actions to support if not already present
        if not row_br_in_support:
            row_support.append(row_br_action)
        if not col_br_in_support:
            col_support.append(col_br_action)
    
    return strategies, supports


def main() -> None:
    pass


if __name__ == '__main__':
    main()
