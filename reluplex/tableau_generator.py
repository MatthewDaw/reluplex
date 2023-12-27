"""Convert problem to standard simplex form."""

from typing import List

import numpy as np
from reluplex.pydantic_dictionaries import ConstraintTypes, ReluConstraint


class TableauFormatter:
    """Convert problem to standard simplex form."""

    def add_not_non_negative_constraint(self, tableau: np.matrix, not_non_negative_constraints:List[int]) -> np.matrix:
        """Substitute neural node variables to allow nerual node variable to be positive or negative values."""
        tableua_columns = []
        for variable_count in range(tableau.shape[1]):
            tableua_columns.append(tableau[:, variable_count])
            if variable_count in not_non_negative_constraints:
                tableua_columns.append(-1 * tableau[:, variable_count])
        tableau = np.matrix(np.hstack(tableua_columns))
        return tableau

    def add_semi_relu_constraints(self, tableau: np.matrix, right_hand_side: np.ndarray,
                      relu_constraints: List[ReluConstraint],
                      constraint_types: List[ConstraintTypes]) -> (np.matrix, np.ndarray, List[ConstraintTypes]):
        """
        Add semi relu constraints to initial problem.

        Semi relu constraints are not in the original reluplex paper. However, we add them anyway as they show to improve
        peroformance. They constrain the positive side of the node before the relu to be equal to the node after the relu.
        This is not equivelent to a relu constraint as it makes it possible for the tableau to have non-zero values for the
        positive and negative values of the node before the relu.
        """
        new_rows_for_tableau = []
        new_right_hand_side_elements = []
        for individual_relu_constraints in relu_constraints:
            new_row = np.zeros(tableau.shape[1])
            new_row[individual_relu_constraints.input_node.positions_in_flat_array[0]] = 1
            new_row[individual_relu_constraints.output_node.positions_in_flat_array[0]] = -1
            new_right_hand_side_elements.append(0)
            constraint_types.append(ConstraintTypes.EQUALITY.value)
            new_rows_for_tableau.append(new_row)
        tableau = np.vstack([tableau] + list(new_rows_for_tableau))
        right_hand_side = np.concatenate([right_hand_side] + [np.array(new_right_hand_side_elements)])
        return tableau, right_hand_side, constraint_types

    def generate_auxilary_variables(self, right_hand_side: np.ndarray, constraint_types: List[ConstraintTypes]) -> np.ndarray:
        """Generate auxilary variables for initial problem."""
        equality_constraints = [
            constraint_count
            for constraint_count, individual_constraint in enumerate(constraint_types)
            if individual_constraint == ConstraintTypes.EQUALITY.value
        ]
        less_than_zero_constraint = list(np.where(right_hand_side < 0)[0])
        all_constraints = equality_constraints + less_than_zero_constraint
        num_constraints = len(set(all_constraints))
        auxilary_columns = np.zeros((len(constraint_types), num_constraints))
        covered_zeros = 0
        for row_count, (individual_constraint, individual_right_hand_side) in enumerate(
            zip(constraint_types, right_hand_side)
        ):
            if individual_right_hand_side < 0 or individual_constraint == ConstraintTypes.EQUALITY.value:
                auxilary_columns[row_count, covered_zeros] = 1
                covered_zeros += 1
        return auxilary_columns

    def generate_slack_variables(self, constraint_types: List[ConstraintTypes]) -> np.ndarray:
        """Generate slack variables."""
        num_inequality_constraints = sum(
            1 for individual_constraint in constraint_types if individual_constraint != ConstraintTypes.EQUALITY.value
        )
        slack_columns = np.zeros((len(constraint_types), num_inequality_constraints))
        covered_slack_variables = 0
        for row_count, individual_constraint in enumerate(constraint_types):
            if individual_constraint != ConstraintTypes.EQUALITY.value:
                slack_columns[row_count, covered_slack_variables] = 1
                covered_slack_variables += 1
        return slack_columns

    def flip_negative_signs(self, full_tableau: np.matrix, right_hand_side: np.ndarray) -> (np.matrix, np.ndarray):
        """Flip negative signs."""
        for row_count in range(len(right_hand_side)):
            if right_hand_side[row_count] < 0:
                right_hand_side[row_count] *= -1
                full_tableau[row_count] *= -1
        return full_tableau, right_hand_side

    def convert_problem_to_standard_form(
        self,
        tableau_without_rhs: np.matrix,
        right_hand_column: np.ndarray,
        not_non_negative_constraints: List[int],
        constraint_types: List[ConstraintTypes],
        relu_constraints: List[ReluConstraint],
    ) -> (np.matrix, np.ndarray, int, int, int):
        """Convert problem to standard simplex form."""
        tableau_without_rhs = self.add_not_non_negative_constraint(tableau_without_rhs, not_non_negative_constraints)
        tableau_without_rhs, right_hand_column, constraint_types = self.add_semi_relu_constraints(
            tableau_without_rhs, right_hand_column, relu_constraints, constraint_types
        )
        aux_columns = self.generate_auxilary_variables(right_hand_column, constraint_types)
        slack_columns = self.generate_slack_variables(constraint_types)
        num_variables = tableau_without_rhs.shape[1]
        num_slack_variables = slack_columns.shape[1]
        num_aux_variables = aux_columns.shape[1]
        full_tableau = np.matrix(np.hstack([tableau_without_rhs, slack_columns]))
        full_tableau, right_hand_column = self.flip_negative_signs(full_tableau, right_hand_column)
        full_tableau = np.hstack([full_tableau, aux_columns])
        full_tableau = np.vstack([full_tableau.T, right_hand_column]).T
        full_c = np.hstack(
            [
                np.zeros(tableau_without_rhs.shape[1]),
                np.zeros(slack_columns.shape[1]),
                -1 * np.ones(aux_columns.shape[1]),
                np.zeros(1),
            ]
        )
        if np.min(aux_columns[:, -1]) == np.max(aux_columns[:-1]) == 0:
            full_c[-1] = 0
        return full_tableau, full_c, num_variables, num_slack_variables, num_aux_variables
