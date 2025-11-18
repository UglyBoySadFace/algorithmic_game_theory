#!/usr/bin/env python3

import sys
import os
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np


def regret_matching(regrets: np.ndarray) -> np.ndarray:
    """Generate a strategy based on the given cumulative regrets.

    Parameters
    ----------
    regrets : np.ndarray
        The vector containing cumulative regret of each action

    Returns
    -------
    np.ndarray
        The generated strategy
    """
    # Regret matching: strategy is proportional to positive regrets
    # If all regrets are non-positive, use uniform distribution
    
    positive_regrets = np.maximum(regrets, 0.0)
    sum_positive_regrets = np.sum(positive_regrets)
    
    if sum_positive_regrets > 0:
        # Strategy is proportional to positive regrets
        strategy = positive_regrets / sum_positive_regrets
    else:
        # Uniform distribution if no positive regrets
        strategy = np.ones(len(regrets), dtype=np.float64) / len(regrets)
    
    return strategy


def regret_minimization(
    row_matrix: np.ndarray, col_matrix: np.ndarray, num_iters: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run Regret Minimization for a given number of iterations.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    num_iters : int
        The number of iterations to run the algorithm for

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        The sequence of `num_iters` average strategy profiles produced by the algorithm
    """
    num_rows, num_cols = row_matrix.shape
    
    # Initialize cumulative regrets to zero
    row_regrets = np.zeros(num_rows, dtype=np.float64)
    col_regrets = np.zeros(num_cols, dtype=np.float64)
    
    # Initialize cumulative strategy sums
    row_strategy_sum = np.zeros(num_rows, dtype=np.float64)
    col_strategy_sum = np.zeros(num_cols, dtype=np.float64)
    
    # List to store average strategies at each iteration
    average_strategies = []
    
    for t in range(1, num_iters + 1):
        # Generate strategies using regret matching
        row_strategy = regret_matching(row_regrets)
        col_strategy = regret_matching(col_regrets)
        
        # Accumulate strategies
        row_strategy_sum += row_strategy
        col_strategy_sum += col_strategy
        
        # Compute average strategies so far
        avg_row_strategy = row_strategy_sum / t
        avg_col_strategy = col_strategy_sum / t
        
        # Store the average strategy profile
        average_strategies.append((avg_row_strategy.copy(), avg_col_strategy.copy()))
        
        # Compute utilities for each action
        # Row player's utility for each action against col_strategy
        row_action_utilities = np.dot(row_matrix, col_strategy)
        
        # Col player's utility for each action against row_strategy
        col_action_utilities = np.dot(col_matrix.T, row_strategy)
        
        # Compute current utility
        row_current_utility = np.dot(row_strategy, row_action_utilities)
        col_current_utility = np.dot(col_strategy, col_action_utilities)
        
        # Update regrets: regret = utility(action) - utility(current strategy)
        row_regrets += row_action_utilities - row_current_utility
        col_regrets += col_action_utilities - col_current_utility
    
    return average_strategies


def main() -> None:
    pass


if __name__ == '__main__':
    main()
