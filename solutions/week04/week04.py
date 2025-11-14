#!/usr/bin/env python3

import sys
import os
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
from scipy.optimize import linprog


def find_nash_equilibrium(row_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find a Nash equilibrium in a zero-sum normal-form game using linear programming.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A strategy profile that forms a Nash equilibrium
    """
    num_rows, num_cols = row_matrix.shape
    
    # For zero-sum games, we can solve for the row player's maximin strategy
    # using linear programming. The column player's strategy is the minimax.
    
    # Row player's LP:
    # Variables: [p_1, ..., p_m, v] where p_i are probabilities and v is the value
    # Maximize v subject to:
    #   For each column j: sum_i(p_i * A[i,j]) >= v
    #   sum_i(p_i) = 1
    #   p_i >= 0
    
    # Convert to minimization: minimize -v
    num_vars = num_rows + 1  # probabilities + value variable
    
    # Objective: minimize -v (last variable)
    c = np.zeros(num_vars, dtype=np.float64)
    c[-1] = -1.0  # We want to maximize v, so minimize -v
    
    # Inequality constraints: -sum_i(p_i * A[i,j]) + v <= 0 for each column j
    # or equivalently: sum_i(p_i * A[i,j]) >= v
    A_ub = []
    b_ub = []
    
    for j in range(num_cols):
        constraint = np.zeros(num_vars, dtype=np.float64)
        for i in range(num_rows):
            constraint[i] = -row_matrix[i, j]  # Negative because of <= form
        constraint[-1] = 1.0  # +v term
        A_ub.append(constraint)
        b_ub.append(0.0)
    
    A_ub = np.array(A_ub, dtype=np.float64)
    b_ub = np.array(b_ub, dtype=np.float64)
    
    # Equality constraint: sum of probabilities = 1
    A_eq = np.zeros((1, num_vars), dtype=np.float64)
    A_eq[0, :num_rows] = 1.0
    b_eq = np.array([1.0], dtype=np.float64)
    
    # Bounds: probabilities in [0, 1], value unbounded
    bounds = [(0.0, 1.0) for _ in range(num_rows)] + [(None, None)]
    
    # Solve the LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if not result.success:
        raise ValueError("Linear programming failed to find a solution")
    
    row_strategy = result.x[:num_rows]
    
    # Now solve for column player's minimax strategy (minimize their maximum loss)
    # Column player's LP:
    # Variables: [q_1, ..., q_n, u] where q_j are probabilities and u is the value
    # Minimize u subject to:
    #   For each row i: sum_j(q_j * A[i,j]) <= u
    #   sum_j(q_j) = 1
    #   q_j >= 0
    
    num_vars_col = num_cols + 1
    
    # Objective: minimize u
    c_col = np.zeros(num_vars_col, dtype=np.float64)
    c_col[-1] = 1.0
    
    # Inequality constraints: sum_j(q_j * A[i,j]) - u <= 0 for each row i
    A_ub_col = []
    b_ub_col = []
    
    for i in range(num_rows):
        constraint = np.zeros(num_vars_col, dtype=np.float64)
        for j in range(num_cols):
            constraint[j] = row_matrix[i, j]
        constraint[-1] = -1.0  # -u term
        A_ub_col.append(constraint)
        b_ub_col.append(0.0)
    
    A_ub_col = np.array(A_ub_col, dtype=np.float64)
    b_ub_col = np.array(b_ub_col, dtype=np.float64)
    
    # Equality constraint: sum of probabilities = 1
    A_eq_col = np.zeros((1, num_vars_col), dtype=np.float64)
    A_eq_col[0, :num_cols] = 1.0
    b_eq_col = np.array([1.0], dtype=np.float64)
    
    # Bounds: probabilities in [0, 1], value unbounded
    bounds_col = [(0.0, 1.0) for _ in range(num_cols)] + [(None, None)]
    
    # Solve the LP
    result_col = linprog(c_col, A_ub=A_ub_col, b_ub=b_ub_col, A_eq=A_eq_col, b_eq=b_eq_col, bounds=bounds_col, method='highs')
    
    if not result_col.success:
        raise ValueError("Linear programming failed to find a solution for column player")
    
    col_strategy = result_col.x[:num_cols]
    
    return row_strategy.astype(np.float64), col_strategy.astype(np.float64)


def find_correlated_equilibrium(row_matrix: np.ndarray, col_matrix: np.ndarray) -> np.ndarray:
    """Find a correlated equilibrium in a normal-form game using linear programming.

    While the cost vector could be selected to optimize a particular objective, such as
    maximizing the sum of players' utilities, the reference solution sets it to the zero
    vector to ensure reproducibility during testing.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix

    Returns
    -------
    np.ndarray
        A distribution over joint actions that forms a correlated equilibrium
    """
    num_rows, num_cols = row_matrix.shape
    num_joint_actions = num_rows * num_cols
    
    # Variables: p[i,j] for each joint action (i,j), flattened
    # We need to find a distribution over joint actions that satisfies:
    # 1. For each row player action i and deviation i':
    #    sum_j p[i,j] * (U_row[i,j] - U_row[i',j]) >= 0
    # 2. For each column player action j and deviation j':
    #    sum_i p[i,j] * (U_col[i,j] - U_col[i,j']) >= 0
    # 3. sum_{i,j} p[i,j] = 1
    # 4. p[i,j] >= 0 for all i,j
    
    # Objective: minimize 0 (just find a feasible solution)
    c = np.zeros(num_joint_actions, dtype=np.float64)
    
    # Inequality constraints for incentive compatibility
    A_ub = []
    b_ub = []
    
    # Row player's incentive constraints
    for i in range(num_rows):
        for i_prime in range(num_rows):
            if i != i_prime:
                # sum_j p[i,j] * (U_row[i,j] - U_row[i',j]) >= 0
                # Rewrite as: -sum_j p[i,j] * (U_row[i,j] - U_row[i',j]) <= 0
                constraint = np.zeros(num_joint_actions, dtype=np.float64)
                for j in range(num_cols):
                    idx = i * num_cols + j
                    constraint[idx] = -(row_matrix[i, j] - row_matrix[i_prime, j])
                A_ub.append(constraint)
                b_ub.append(0.0)
    
    # Column player's incentive constraints
    for j in range(num_cols):
        for j_prime in range(num_cols):
            if j != j_prime:
                # sum_i p[i,j] * (U_col[i,j] - U_col[i,j']) >= 0
                # Rewrite as: -sum_i p[i,j] * (U_col[i,j] - U_col[i,j']) <= 0
                constraint = np.zeros(num_joint_actions, dtype=np.float64)
                for i in range(num_rows):
                    idx = i * num_cols + j
                    constraint[idx] = -(col_matrix[i, j] - col_matrix[i, j_prime])
                A_ub.append(constraint)
                b_ub.append(0.0)
    
    A_ub = np.array(A_ub, dtype=np.float64) if A_ub else None
    b_ub = np.array(b_ub, dtype=np.float64) if b_ub else None
    
    # Equality constraint: probabilities sum to 1
    A_eq = np.ones((1, num_joint_actions), dtype=np.float64)
    b_eq = np.array([1.0], dtype=np.float64)
    
    # Bounds: all probabilities in [0, 1]
    bounds = [(0.0, 1.0) for _ in range(num_joint_actions)]
    
    # Solve the LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if not result.success:
        raise ValueError("Linear programming failed to find a correlated equilibrium")
    
    # Reshape the distribution back to matrix form
    distribution = result.x.reshape((num_rows, num_cols))
    
    return distribution.astype(np.float64)


def main() -> None:
    pass


if __name__ == '__main__':
    main()
