#!/usr/bin/env python3


import sys
import os
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from itertools import combinations
from solutions.week01 import week01


def plot_best_response_value_function(row_matrix: np.ndarray, step_size: float) -> None:
    """Plot the best response value function for the row player in a 2xN zero-sum game.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    step_size : float
        The step size for the probability of the first action of the row player
    """
    if row_matrix.shape[0] != 2:
        raise ValueError("This function is designed for 2xN games only")
    
    # Generate probabilities for the first action (p1)
    # The probability for the second action is (1 - p1)
    p1_values = np.arange(0, 1 + step_size, step_size)
    
    # For each probability, compute the best response value
    best_response_values = []
    
    for p1 in p1_values:
        p2 = 1 - p1
        row_strategy = np.array([p1, p2])
        
        # For each column action, compute expected payoff for column player
        # In zero-sum game, column player wants to minimize row player's payoff
        col_payoffs = []
        for j in range(row_matrix.shape[1]):
            # Expected payoff when column player plays pure strategy j
            expected_payoff = np.sum(row_strategy * row_matrix[:, j])
            col_payoffs.append(expected_payoff)
        
        # Column player's best response is the action that minimizes row player's payoff
        best_response_value = min(col_payoffs)
        best_response_values.append(best_response_value)
    
    # Plot the best response value function
    plt.figure(figsize=(10, 6))
    plt.plot(p1_values, best_response_values, 'b-', linewidth=2)
    plt.xlabel('Probability of Row Player\'s First Action')
    plt.ylabel('Best Response Value')
    plt.title('Best Response Value Function for Row Player (2xN Zero-Sum Game)')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.savefig('./best_response_value_function.png')


import numpy as np
from scipy.optimize import linprog


def verify_support(
    matrix: np.ndarray, row_support: np.ndarray, col_support: np.ndarray
) -> np.ndarray | None:
    """Construct a system of linear equations to check whether there
    exists a candidate for a Nash equilibrium for the given supports.
    
    The reference implementation uses `scipy.optimize.linprog`
    with the default solver -- 'highs'. You can find more information at
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
    
    Parameters
    ----------
    matrix : np.ndarray
        A payoff matrix of one of the players (row player's payoffs)
    row_support : np.ndarray
        Array of indices indicating the row player's support
    col_support : np.ndarray
        Array of indices indicating the column player's support
    
    Returns
    -------
    np.ndarray | None
        The column player's strategy (full vector), if it exists, otherwise `None`
    """
    n_rows, n_cols = matrix.shape
    
    # We're looking for a column player's strategy such that:
    # 1. All row actions in row_support get the same expected payoff
    # 2. All row actions NOT in row_support get at most that payoff
    # 3. Column player only plays actions in col_support
    
    # Variables: [q_1, q_2, ..., q_m, u]
    # where q_j are probabilities for columns in col_support, u is the common payoff
    n_support = len(col_support)
    n_vars = n_support + 1
    
    if n_support == 0:
        return None
    
    # Objective: minimize 0 (feasibility problem)
    c = np.zeros(n_vars)
    
    # === Equality Constraints ===
    A_eq = []
    b_eq = []
    
    # 1. Probabilities sum to 1
    prob_sum = np.zeros(n_vars)
    prob_sum[:n_support] = 1
    A_eq.append(prob_sum)
    b_eq.append(1)
    
    # 2. All actions in row_support must yield the same expected payoff u
    # For each row i in row_support: sum_j matrix[i,j] * q_j = u
    # Rewritten: sum_j matrix[i,j] * q_j - u = 0
    for i in row_support:
        constraint = np.zeros(n_vars)
        for idx, j in enumerate(col_support):
            constraint[idx] = matrix[i, j]
        constraint[-1] = -1  # -u
        A_eq.append(constraint)
        b_eq.append(0)
    
    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)
    
    # === Inequality Constraints ===
    A_ub = []
    b_ub = []
    
    # Actions NOT in row_support must yield payoff <= u
    # For row i not in support: sum_j matrix[i,j] * q_j <= u
    # Rewritten: sum_j matrix[i,j] * q_j - u <= 0
    row_support_set = set(row_support)
    for i in range(n_rows):
        if i not in row_support_set:
            constraint = np.zeros(n_vars)
            for idx, j in enumerate(col_support):
                constraint[idx] = matrix[i, j]
            constraint[-1] = -1  # -u
            A_ub.append(constraint)
            b_ub.append(0)
    
    if len(A_ub) > 0:
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
    else:
        A_ub = None
        b_ub = None
    
    # Bounds: probabilities in [0, 1], utility unbounded
    bounds = [(0, 1) for _ in range(n_support)] + [(None, None)]
    
    # Solve
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                     bounds=bounds, method='highs')
    
    if not result.success:
        return None
    
    # Extract probabilities and construct full strategy vector
    probs = result.x[:n_support]
    
    # Create full column strategy (length = number of columns)
    full_strategy = np.zeros(n_cols, dtype=np.float64)
    for idx, j in enumerate(col_support):
        full_strategy[j] = probs[idx]
    
    return full_strategy


def support_enumeration(
    row_matrix: np.ndarray, col_matrix: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run the Support Enumeration algorithm and return a list of all Nash equilibria

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        A list of strategy profiles corresponding to found Nash equilibria
    """
    num_rows, num_cols = row_matrix.shape
    nash_equilibria = []
    
    # Enumerate all possible support pairs
    for row_support_size in range(1, num_rows + 1):
        for col_support_size in range(1, num_cols + 1):
            
            # Generate all combinations of supports of given sizes
            for row_support in combinations(range(num_rows), row_support_size):
                for col_support in combinations(range(num_cols), col_support_size):
                    
                    row_support = np.array(row_support)
                    col_support = np.array(col_support)
                    
                    # Try to find column strategy given row support
                    col_strategy = verify_support(row_matrix, row_support, col_support)
                    
                    if col_strategy is not None:
                        # Try to find row strategy given column support
                        # For this, we use the transpose of col_matrix
                        row_strategy = verify_support(col_matrix.T, col_support, row_support)
                        
                        if row_strategy is not None:
                            # Verify this is actually a Nash equilibrium
                            if is_nash_equilibrium(row_matrix, col_matrix, row_strategy, col_strategy):
                                # Check if we already found this equilibrium (avoid duplicates)
                                is_duplicate = False
                                for existing_row, existing_col in nash_equilibria:
                                    if (np.allclose(row_strategy, existing_row, atol=1e-6) and 
                                        np.allclose(col_strategy, existing_col, atol=1e-6)):
                                        is_duplicate = True
                                        break
                                
                                if not is_duplicate:
                                    nash_equilibria.append((row_strategy, col_strategy))
    
    return nash_equilibria


def is_nash_equilibrium(row_matrix: np.ndarray, col_matrix: np.ndarray, 
                       row_strategy: np.ndarray, col_strategy: np.ndarray, 
                       tolerance: float = 1e-6) -> bool:
    """Check if a strategy profile is a Nash equilibrium"""
    
    # Check if row player has any profitable deviation
    current_row_utility = np.dot(row_strategy, np.dot(row_matrix, col_strategy))
    
    for i in range(row_matrix.shape[0]):
        pure_strategy = np.zeros(row_matrix.shape[0])
        pure_strategy[i] = 1.0
        deviation_utility = np.dot(pure_strategy, np.dot(row_matrix, col_strategy))
        
        if deviation_utility > current_row_utility + tolerance:
            return False
    
    # Check if column player has any profitable deviation
    current_col_utility = np.dot(row_strategy, np.dot(col_matrix, col_strategy))
    
    for j in range(col_matrix.shape[1]):
        pure_strategy = np.zeros(col_matrix.shape[1])
        pure_strategy[j] = 1.0
        deviation_utility = np.dot(row_strategy, np.dot(col_matrix, pure_strategy))
        
        if deviation_utility > current_col_utility + tolerance:
            return False
    
    return True


def main() -> None:
    # Example usage
    
    # Example 1: Plot best response value function for a simple 2x3 game
    print("Example 1: Best Response Value Function")
    row_matrix = np.array([[3, 1, 4], [2, 5, 1]], dtype=np.float32)
    plot_best_response_value_function(row_matrix, 0.01)
    
    # Example 2: Support enumeration for a simple 2x2 game
    print("\nExample 2: Support Enumeration")
    row_matrix = np.array([[3, 0], [0, 2]], dtype=np.float32)
    col_matrix = np.array([[2, 0], [0, 3]], dtype=np.float32)
    
    equilibria = support_enumeration(row_matrix, col_matrix)
    print(f"Found {len(equilibria)} Nash equilibria:")
    for i, (row_strat, col_strat) in enumerate(equilibria):
        print(f"Equilibrium {i+1}:")
        print(f"  Row strategy: {row_strat}")
        print(f"  Col strategy: {col_strat}")


if __name__ == '__main__':
    main()
