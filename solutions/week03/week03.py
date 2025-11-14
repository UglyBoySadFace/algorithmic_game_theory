#!/usr/bin/env python3

import sys
import os
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from solutions.week01 import week01


def compute_deltas(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    row_strategy: np.ndarray,
    col_strategy: np.ndarray,
) -> np.ndarray:
    """Compute players' incentives to deviate from their strategies.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.ndarray
        Each player's incentive to deviate
    """
    # Current utility for each player
    current_row_utility = np.dot(row_strategy, np.dot(row_matrix, col_strategy))
    current_col_utility = np.dot(row_strategy, np.dot(col_matrix, col_strategy))
    
    # Best response utilities
    row_best_response = week01.calculate_best_response_against_col(row_matrix, col_strategy)
    col_best_response = week01.calculate_best_response_against_row(col_matrix, row_strategy)
    
    best_row_utility = np.dot(row_best_response, np.dot(row_matrix, col_strategy))
    best_col_utility = np.dot(row_strategy, np.dot(col_matrix, col_best_response))
    
    # Delta is the gain from deviating to best response
    row_delta = best_row_utility - current_row_utility
    col_delta = best_col_utility - current_col_utility
    
    return np.array([row_delta, col_delta], dtype=np.float64)


def compute_nash_conv(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    row_strategy: np.ndarray,
    col_strategy: np.ndarray,
) -> np.float64:
    """Compute the NashConv value of a given strategy profile.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.float64
        The NashConv value of the given strategy profile
    """
    # NashConv is the sum of both players' incentives to deviate
    deltas = compute_deltas(row_matrix, col_matrix, row_strategy, col_strategy)
    return np.float64(np.sum(deltas))


def compute_exploitability(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    row_strategy: np.ndarray,
    col_strategy: np.ndarray,
) -> np.float64:
    """Compute the exploitability of a given strategy profile.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.float64
        The exploitability value of the given strategy profile
    """
    # Exploitability is the average of both players' incentives to deviate
    deltas = compute_deltas(row_matrix, col_matrix, row_strategy, col_strategy)
    return np.float64(np.mean(deltas))


def fictitious_play(
    row_matrix: np.ndarray, col_matrix: np.ndarray, num_iters: int, naive: bool
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run Fictitious Play for a given number of iterations.

    Although any averaging method is valid, the reference solution updates the
    average strategy vectors using a moving average. Therefore, it is recommended
    to use the same averaging method to avoid numerical discrepancies during testing.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    num_iters : int
        The number of iterations to run the algorithm for
    naive : bool
        Whether to calculate the best response against the last
        opponent's strategy or the average opponent's strategy

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        The sequence of average strategy profiles produced by the algorithm
    """
    num_row_actions = row_matrix.shape[0]
    num_col_actions = row_matrix.shape[1]
    
    # Initialize with uniform random strategies
    row_strategy = np.ones(num_row_actions, dtype=np.float64) / num_row_actions
    col_strategy = np.ones(num_col_actions, dtype=np.float64) / num_col_actions
    
    # Keep track of average strategies
    row_avg_strategy = row_strategy.copy()
    col_avg_strategy = col_strategy.copy()
    
    # Store the sequence of average strategy profiles
    strategies = []
    
    for t in range(1, num_iters + 1):
        # Decide which strategy to best-respond to
        if naive:
            # Naive: best respond to the last strategy
            row_target = row_strategy
            col_target = col_strategy
        else:
            # Standard: best respond to the average strategy
            row_target = row_avg_strategy
            col_target = col_avg_strategy
        
        # Compute best responses
        row_best_response = week01.calculate_best_response_against_col(row_matrix, col_target)
        col_best_response = week01.calculate_best_response_against_row(col_matrix, row_target)
        
        # Update current strategies to best responses
        row_strategy = row_best_response
        col_strategy = col_best_response
        
        # Update average strategies using moving average
        # avg_{t} = (1/t) * current + ((t-1)/t) * avg_{t-1}
        row_avg_strategy = (row_strategy + (t - 1) * row_avg_strategy) / t
        col_avg_strategy = (col_strategy + (t - 1) * col_avg_strategy) / t
        
        # Store the current average strategy profile
        strategies.append((row_avg_strategy.copy(), col_avg_strategy.copy()))
    
    return strategies


def plot_exploitability(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    strategies: list[tuple[np.ndarray, np.ndarray]],
    label: str,
) -> list[np.float64]:
    """Compute and plot the exploitability of a sequence of strategy profiles.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    strategies : list[tuple[np.ndarray, np.ndarray]]
        The sequence of strategy profiles
    label : str
        The name of the algorithm that produced `strategies`

    Returns
    -------
    list[np.float64]
        A sequence of exploitability values, one for each strategy profile
    """
    exploitabilities = []
    
    for row_strategy, col_strategy in strategies:
        exploit = compute_exploitability(row_matrix, col_matrix, row_strategy, col_strategy)
        exploitabilities.append(exploit)
    
    # Plot the exploitability over time
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(exploitabilities) + 1), exploitabilities, label=label, linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Exploitability')
    plt.title('Exploitability Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'./exploitability_{label.replace(" ", "_").lower()}.png')
    
    return exploitabilities


def main() -> None:
    """Main function to run Fictitious Play and plot exploitability."""
    # Resource matrix for Rock-Paper-Scissors
    row_matrix = np.array([[0, -1, 1],
                           [1, 0, -1],
                           [-1, 1, 0]], dtype=np.float64)
    col_matrix = -row_matrix.T  # Zero-sum game
    
    num_iters = 100
    
    # Run standard Fictitious Play
    standard_strategies = fictitious_play(row_matrix, col_matrix, num_iters, naive=False)
    plot_exploitability(row_matrix, col_matrix, standard_strategies, label='Standard Fictitious Play')
    
    # Run naive Fictitious Play
    naive_strategies = fictitious_play(row_matrix, col_matrix, num_iters, naive=True)
    plot_exploitability(row_matrix, col_matrix, naive_strategies, label='Naive Fictitious Play')


if __name__ == '__main__':
    main()
