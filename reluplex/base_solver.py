"""Base solver for solving standardized simplex problem."""
import numpy as np
from typing import List

class BaseSolver:
    """Base solver for solving standardized simplex problem."""

    def perform_pivot(self, tableau: np.matrix, col: int, row: int) -> np.matrix:
        """Perform simplex pivot."""
        pri_mult = tableau[row, col]
        tableau[row] /= pri_mult
        for i in range(len(tableau)):
            if i != row:
                sec_mult = tableau[i, col]  # secondary multiplier from row being updated
                pri_row = tableau[row] * sec_mult
                tableau[i] -= pri_row
        return tableau

    def get_pivot_row(self, tableau: np.matrix, pivot_column: int) -> int:
        """Determine what row to pivot by given column number."""
        arguments_to_consider = np.where(tableau[:-1, pivot_column] > 0)
        if len(arguments_to_consider[0]) == 0:
            return None
        ratios = np.argsort(
            (tableau[arguments_to_consider[0], -1] / tableau[arguments_to_consider[0], pivot_column]).flatten()
        )
        minimum_ratio = ratios[0, 0]
        pivot_row = int(arguments_to_consider[0][minimum_ratio])
        return pivot_row

    def base_solve(self, tableau: np.matrix, slack_values: List[int], minimize: bool) -> (np.matrix, List[int], bool):
        """Solve standardized simplex problem."""
        solution_is_feasible = True
        if minimize:
            while np.max(tableau[-1, :-1]) > 0:
                pivot_column = int(np.argmax(tableau[-1, :-1]))
                pivot_row = self.get_pivot_row(tableau, pivot_column)
                if pivot_row is None:
                    solution_is_feasible = False
                    return tableau, slack_values, solution_is_feasible
                tableau = self.perform_pivot(tableau, pivot_column, pivot_row)
                slack_values[pivot_row] = pivot_column
        else:
            while np.min(tableau[-1, :-1]) < 0:
                pivot_column = int(np.argmin(tableau[-1, :-1]))
                pivot_row = self.get_pivot_row(tableau, pivot_column)
                if pivot_row is None:
                    solution_is_feasible = False
                    return tableau, slack_values, solution_is_feasible
                tableau = self.perform_pivot(tableau, pivot_column, pivot_row)
                slack_values[pivot_row] = pivot_column
        return tableau, slack_values, solution_is_feasible
