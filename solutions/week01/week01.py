#!/usr/bin/env python3

import numpy as np


def evaluate_general_sum(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    row_strategy: np.ndarray,
    col_strategy: np.ndarray,
) -> np.ndarray:
    """Compute the expected utility of each player in a general-sum game.

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
        A vector of expected utilities of the players
    """
    # Expected utility for row player: row_strategy^T * row_matrix * col_strategy
    row_utility = np.dot(row_strategy, np.dot(row_matrix, col_strategy))
    
    # Expected utility for column player: row_strategy^T * col_matrix * col_strategy
    col_utility = np.dot(row_strategy, np.dot(col_matrix, col_strategy))
    
    return np.array([row_utility, col_utility], dtype=np.float64)


def evaluate_zero_sum(
    row_matrix: np.ndarray, row_strategy: np.ndarray, col_strategy: np.ndarray
) -> np.ndarray:
    """Compute the expected utility of each player in a zero-sum game.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.ndarray
        A vector of expected utilities of the players
    """
    # In zero-sum games, column player's payoff matrix is -row_matrix
    # Row player utility: row_strategy^T * row_matrix * col_strategy
    row_utility = np.dot(row_strategy, np.dot(row_matrix, col_strategy))
    
    # Column player utility is the negative of row player utility
    col_utility = -row_utility
    
    return np.array([row_utility, col_utility], dtype=np.float64)


def calculate_best_response_against_row(
    col_matrix: np.ndarray, row_strategy: np.ndarray
) -> np.ndarray:
    """Compute a pure best response for the column player against the row player.

    Parameters
    ----------
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy

    Returns
    -------
    np.ndarray
        The column player's best response
    """
    # Calculate expected payoffs for each column action
    # Expected payoff for action j: sum_i(row_strategy[i] * col_matrix[i,j])
    expected_payoffs = np.dot(row_strategy, col_matrix)
    
    # Find the action with maximum expected payoff
    best_action = np.argmax(expected_payoffs)
    
    # Create pure strategy (one-hot vector)
    best_response = np.zeros(col_matrix.shape[1], dtype=np.float64)
    best_response[best_action] = 1.0
    
    return best_response


def calculate_best_response_against_col(
    row_matrix: np.ndarray, col_strategy: np.ndarray
) -> np.ndarray:
    """Compute a pure best response for the row player against the column player.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.ndarray
        The row player's best response
    """
    # Calculate expected payoffs for each row action
    # Expected payoff for action i: sum_j(row_matrix[i,j] * col_strategy[j])
    expected_payoffs = np.dot(row_matrix, col_strategy)
    
    # Find the action with maximum expected payoff
    best_action = np.argmax(expected_payoffs)
    
    # Create pure strategy (one-hot vector)
    best_response = np.zeros(row_matrix.shape[0], dtype=np.float64)
    best_response[best_action] = 1.0
    
    return best_response


def evaluate_row_against_best_response(
    row_matrix: np.ndarray, col_matrix: np.ndarray, row_strategy: np.ndarray
) -> np.float64:
    """Compute the utility of the row player when playing against a best response strategy.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy

    Returns
    -------
    np.float64
        The expected utility of the row player
    """
    # Calculate column player's best response against row strategy
    col_best_response = calculate_best_response_against_row(col_matrix, row_strategy)
    
    # Calculate row player's utility against this best response
    row_utility = np.dot(row_strategy, np.dot(row_matrix, col_best_response))
    
    return np.float64(row_utility)


def evaluate_col_against_best_response(
    row_matrix: np.ndarray, col_matrix: np.ndarray, col_strategy: np.ndarray
) -> np.float64:
    """Compute the utility of the column player when playing against a best response strategy.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.float64
        The expected utility of the column player
    """
    # Calculate row player's best response against column strategy
    row_best_response = calculate_best_response_against_col(row_matrix, col_strategy)
    
    # Calculate column player's utility against this best response
    col_utility = np.dot(row_best_response, np.dot(col_matrix, col_strategy))
    
    return np.float64(col_utility)


def find_strictly_dominated_actions(matrix: np.ndarray) -> np.ndarray:
    """Find strictly dominated actions for the given normal-form game.

    Parameters
    ----------
    matrix : np.ndarray
        A payoff matrix of one of the players

    Returns
    -------
    np.ndarray
        Indices of strictly dominated actions
    """
    num_actions = matrix.shape[0]
    dominated_actions = []
    
    for i in range(num_actions):
        for j in range(num_actions):
            if i != j:
                # Check if action i is strictly dominated by action j
                # Action i is dominated by j if matrix[j,:] > matrix[i,:] for all columns
                if np.all(matrix[j, :] > matrix[i, :]):
                    dominated_actions.append(i)
                    break  # Once we find a dominating action, we don't need to check others
    
    return np.array(dominated_actions, dtype=np.int64)


def iterated_removal_of_dominated_strategies(
    row_matrix: np.ndarray, col_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run Iterated Removal of Dominated Strategies.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Four-tuple of reduced row and column payoff matrices, and remaining row and column actions
    """
    # Make copies to avoid modifying original matrices
    current_row_matrix = row_matrix.copy()
    current_col_matrix = col_matrix.copy()
    
    # Keep track of remaining actions
    remaining_row_actions = np.arange(row_matrix.shape[0])
    remaining_col_actions = np.arange(col_matrix.shape[1])
    
    changed = True
    while changed:
        changed = False
        
        # Check for dominated row actions
        row_dominated = find_strictly_dominated_actions(current_row_matrix)
        if len(row_dominated) > 0:
            # Remove dominated rows
            keep_rows = np.array([i for i in range(current_row_matrix.shape[0]) if i not in row_dominated])
            current_row_matrix = current_row_matrix[keep_rows, :]
            current_col_matrix = current_col_matrix[keep_rows, :]
            remaining_row_actions = remaining_row_actions[keep_rows]
            changed = True
        
        # Check for dominated column actions
        # For column player, we need to transpose the matrix to use the same function
        col_dominated = find_strictly_dominated_actions(current_col_matrix.T)
        if len(col_dominated) > 0:
            # Remove dominated columns
            keep_cols = np.array([j for j in range(current_col_matrix.shape[1]) if j not in col_dominated])
            current_row_matrix = current_row_matrix[:, keep_cols]
            current_col_matrix = current_col_matrix[:, keep_cols]
            remaining_col_actions = remaining_col_actions[keep_cols]
            changed = True
    
    return current_row_matrix, current_col_matrix, remaining_row_actions, remaining_col_actions


def main() -> None:
    pass


if __name__ == '__main__':
    main()
