"""ReluPlex Solver."""

import numpy as np
from reluplex.base_solver import BaseSolver
from typing import List

class ReluPlexSolver:
    """ReluPlex solver."""

    def __init__(self):
        self.base_solver = BaseSolver()

    def _generate_feasibility_solve_slack_values(
        self, phase_one_tabelau: np.matrix, num_variables: int, num_slack_variables: int
    ):
        """Generate slack values for determining initial feasible point."""
        slack_values = []
        for row_count in range(phase_one_tabelau.shape[0] - 1):
            non_zero_values = np.where(phase_one_tabelau[row_count, num_variables + num_slack_variables : -1])
            if len(non_zero_values[0]):
                slack_values.append(list(non_zero_values[1])[0] + num_variables + num_slack_variables)
            else:
                non_zero_values = np.where(
                    phase_one_tabelau[row_count, num_variables : num_variables + num_slack_variables]
                )
                slack_values.append(list(non_zero_values[1])[0] + num_variables)
        return slack_values

    def feasibility_solve(self, phase_one_tabelau: np.matrix, num_variables: int, num_slack_variables: int) -> (np.matrix, List[int], bool):
        """Find initial feasible point or determine infeasibility."""
        slack_values = self._generate_feasibility_solve_slack_values(
            phase_one_tabelau, num_variables, num_slack_variables
        )
        aux_columns = np.where(phase_one_tabelau[-1, :])
        for a_col in aux_columns[1]:
            row_pivot = np.where(phase_one_tabelau[:-1, a_col])
            row_pivot = row_pivot[0][0]
            phase_one_tabelau = self.base_solver.perform_pivot(phase_one_tabelau, a_col, row_pivot)
            slack_values[row_pivot] = a_col
        return self.base_solver.base_solve(phase_one_tabelau, slack_values, True)

    def find_primal_col(self, constraint_row: np.matrix, objective_row: np.matrix) -> (int, bool):
        """Find column to pivot on for primal dual problem."""
        candidate_columns = np.where(constraint_row[0, :-1] < 0)[1]
        if len(candidate_columns) == 0:
            return 0, False
        quotionts = np.absolute(objective_row[0, candidate_columns] / constraint_row[0, candidate_columns])
        values_to_consider = np.where(0 < quotionts)
        if 0 < len(values_to_consider[1]):
            quotionts = quotionts[0, values_to_consider[1]]
            argmin = np.argmin(quotionts)
        else:
            return 0, False
        return candidate_columns[argmin], True

    def convert_to_constraint_row(self, tableau: np.matrix, col_to_set_to_zero: int, slack_values: List[int]) -> np.matrix:
        """Convert col_to_set_to_zero to a constraint row that can be added to tableau."""
        constraint_row = np.matrix(np.zeros(tableau.shape[1]))
        constraint_row[0, col_to_set_to_zero] = 1
        if col_to_set_to_zero in slack_values:
            pivot_row = slack_values.index(col_to_set_to_zero)
            constraint_row = constraint_row - (tableau[pivot_row] / tableau[pivot_row, col_to_set_to_zero])
        return constraint_row

    def remove_last_relu_constraint(self, tableau: np.matrix, slack_values: List[int]) -> (np.matrix, List[int]):
        """Remove the last relu constraint that was imposed on the problem setup."""
        smarter_row = self.base_solver.get_pivot_row(tableau, tableau.shape[1] - 2)
        tableau = self.base_solver.perform_pivot(tableau, tableau.shape[1] - 2, smarter_row)
        slack_values[smarter_row] = tableau.shape[1] - 2
        tableau = np.delete(tableau, smarter_row, 0)
        tableau = np.delete(tableau, slack_values[smarter_row], 1)
        del slack_values[smarter_row]
        return tableau, slack_values

    def adjust_value(self, tableau: np.matrix, col_to_set_to_zero: int, slack_values: List[int]) -> (np.matrix, List[int], bool):
        """Adjust value of either the positive or negatie side of a node variable to zero."""
        new_constraint_row = self.convert_to_constraint_row(tableau, col_to_set_to_zero, slack_values)
        tableau = np.insert(tableau, [-1], new_constraint_row, axis=0)
        tableau[-1] = new_constraint_row
        new_aux_column = np.zeros(tableau.shape[0])
        new_aux_column[-2] = 1
        tableau = np.insert(tableau, -1, new_aux_column, axis=1)
        slack_values.append(tableau.shape[1] - 2)
        objective_row = np.zeros(tableau.shape[1])
        objective_row[col_to_set_to_zero] = 1
        tableau[-1, :-1] = -1
        while np.min(tableau[:-1, -1]) < 0:
            row_to_pivot = int(np.argmin(tableau[:-1, -1]))
            column_pivot, problem_is_feasible = self.find_primal_col(tableau[row_to_pivot], tableau[-1])
            if not problem_is_feasible:
                return tableau, slack_values, False
            tableau = self.base_solver.perform_pivot(tableau, column_pivot, row_to_pivot)
            slack_values[row_to_pivot] = column_pivot
        return tableau, slack_values, True
