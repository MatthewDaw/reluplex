"""Main parent file for reluplex object."""
from typing import List, Tuple
import copy
import numpy as np
from reluplex.problem_contextualizer import ProblemContextualizer
from reluplex.solver import ReluPlexSolver
from torch import nn
from reluplex.pydantic_dictionaries import ReluConstraint

class ReluPlex:
    """Reluplex parent object."""

    def __init__(self, sequential_network: nn.Sequential):
        self.problem_contextualizer = ProblemContextualizer(sequential_network)
        self.solver = ReluPlexSolver()

    def add_lower_bound_constraint_to_input(self, node_count: int, bound_to_set: float) -> None:
        """Add lower bound constraint to input."""
        self.problem_contextualizer.add_explicit_constraint(node_count, bound_to_set, True, True)

    def add_upper_bound_constraint_to_input(self, node_count: int, bound_to_set: float) -> None:
        """Add upper bound constraint to input."""
        self.problem_contextualizer.add_explicit_constraint(node_count, bound_to_set, True, False)

    def add_lower_bound_constraint_to_output(self, node_count: int, bound_to_set: float) -> None:
        """Add lower bound constraint to output."""
        self.problem_contextualizer.add_explicit_constraint(node_count, bound_to_set, False, True)

    def add_upper_bound_constraint_to_output(self, node_count: int, bound_to_set: float) -> None:
        """Add upper bound constraint to output."""
        self.problem_contextualizer.add_explicit_constraint(node_count, bound_to_set, False, False)

    def remove_aux_variables(self, tableau: np.matrix, num_aux_variables: int) -> np.matrix:
        """Remove auxiliary variables from tableau."""
        return np.matrix(np.hstack([tableau[:, : -1 - num_aux_variables], tableau[:, -1]]))

    def merge_non_negative_entries(self, values: List[float]) -> List[float]:
        """Merge the positive and negative split variables into combined node values."""
        true_answers = []
        change_pointer = 0
        true_variable_counter = 0
        while change_pointer < len(values):
            if true_variable_counter in self.problem_contextualizer.not_non_negative_constraints:
                true_answers.append(values[change_pointer] - values[change_pointer + 1])
                change_pointer += 2
            else:
                true_answers.append(values[change_pointer])
                change_pointer += 1
            true_variable_counter += 1
        return true_answers

    def read_solution(self, tableau: np.matrix, num_variables: int, slack_values: List[int]) -> (List[float], List[float]):
        """Read solution point from simplex tableau."""
        values = []
        for val in range(num_variables):
            if val in slack_values:
                values.append(tableau[slack_values.index(val), -1])
            else:
                values.append(0)
        return self.merge_non_negative_entries(values), values

    def find_relu_violation_pairs(self, values: List[float]) -> List[Tuple[List[int],ReluConstraint]]:
        """Find relu violation pairs."""
        violation_pairs = []
        for individual_constraint in self.problem_contextualizer.relu_constraints:
            input_nodes = individual_constraint.input_node.positions_in_flat_array
            if values[input_nodes[0]] != 0 and values[input_nodes[1]] != 0:
                violation_pairs.append((input_nodes, individual_constraint))
        return violation_pairs

    def perform_one_side_of_relu_split(
        self, tableau: np.matrix, slack_values: List[int], num_variables: int, attempt_count: List[int], column_to_constrain_to_zero: int
    ) -> (bool, np.matrix, List[int]):
        """Perform left relu split."""
        potential_tableau, potential_slack_values, solution_is_feasible = self.solver.adjust_value(
            tableau, column_to_constrain_to_zero, slack_values
        )
        if solution_is_feasible:
            solution_is_feasible, tableau, slack_values = self.relu_search(
                potential_tableau, potential_slack_values, num_variables, attempt_count
            )
        return solution_is_feasible, tableau, slack_values

    def relu_split(self, tableau: np.matrix, slack_values: List[int], num_variables: int, relavent_position: List[int], attempt_count: List[int]) -> (bool, np.matrix, List[int]):
        """Perform relu split."""
        # TODO: left and right splits should be run in parallel
        reserve_attempt_count = copy.deepcopy(attempt_count)
        left_solution_is_feasible, left_tableau, left_slack_values = self.perform_one_side_of_relu_split(
            tableau, slack_values, num_variables, attempt_count, relavent_position[0]
        )

        if left_solution_is_feasible:
            return left_solution_is_feasible, left_tableau, left_slack_values
        right_solution_is_feasible, right_tableau, right_slack_values = self.perform_one_side_of_relu_split(
            tableau, slack_values, num_variables, reserve_attempt_count, relavent_position[1]
        )
        if right_solution_is_feasible:
            return right_solution_is_feasible, right_tableau, right_slack_values
        return False, None, None

    def attempt_relu_fix(self, relu_violations: List[Tuple[List[int],ReluConstraint]], attempt_count: List[int], tableau: np.matrix, slack_values: List[int]) -> (np.matrix, List[int], bool):
        """Attempt to fix individual relu violation."""
        first_attempt = relu_violations[0][0][0]
        second_attempt = relu_violations[0][0][1]
        if attempt_count[relu_violations[0][1].constraint_count] % 2 == 1:
            first_attempt = relu_violations[0][0][1]
            second_attempt = relu_violations[0][0][0]
        # successfully imposing these constraints without making the solution infeasible fixes the relu violation.
        potential_tableau, potential_slack_values, solution_is_feasible = self.solver.adjust_value(
            tableau, first_attempt, slack_values
        )
        if not solution_is_feasible:
            # if constraining the positive variable to zero fails, we try constraining the negative variable to zero.
            potential_tableau, potential_slack_values, solution_is_feasible = self.solver.adjust_value(
                tableau, second_attempt, slack_values
            )
        return potential_tableau, potential_slack_values, solution_is_feasible

    def relu_search(self, tableau: np.matrix, slack_values: List[int], num_variables: int, attempt_count: List[int]):
        """Search for relu violations fixes."""
        # declair solution to point to be feasible (though it may not follow relu constraints)
        solution_is_feasible = True
        # First check if there are relu violations.
        _, values = self.read_solution(tableau, num_variables, slack_values)
        relu_violations = self.find_relu_violation_pairs(values)
        # if there are relu violations, try to fix them.
        if 0 < len(relu_violations):
            # continue fixing until there are no more violations or we prove the solution is infeasible.
            while 0 < len(relu_violations) and solution_is_feasible:
                attempt_count[relu_violations[0][1].constraint_count] += 1
                # if we have tried to fix the same violation 3 times, we perform relusplit so that problem is more likely to converge.
                if 3 < max(attempt_count):
                    specific_violation = self.problem_contextualizer.relu_constraints[np.argmax(attempt_count)]
                    relavent_position = specific_violation.input_node.positions_in_flat_array
                    return self.relu_split(tableau, slack_values, num_variables, relavent_position, attempt_count)

                updated_tableau, updated_slack_values, solution_is_feasible = self.attempt_relu_fix(
                    relu_violations, attempt_count, tableau, slack_values
                )
                # if the left or right constraint work, then we update A and check for other relu violations.
                if solution_is_feasible:
                    tableau = updated_tableau
                    slack_values = updated_slack_values
                    _, values = self.read_solution(tableau, num_variables, slack_values)
                    relu_violations = self.find_relu_violation_pairs(values)
        return solution_is_feasible, tableau, slack_values

    def find_feasible_point(self):
        """Find feasible point with constraints or determine if setup is impossible."""
        # initialize simplex tableau
        (
            full_tableau,
            full_c,
            num_variables,
            num_slack_variables,
            num_aux_variables,
        ) = self.problem_contextualizer.convert_to_standard_simplex_form()
        phase_one_tableau = np.matrix(np.vstack([full_tableau, full_c]))
        # determine if a feasible point exists even without relu constraints
        tableau, slack_values, solution_is_feasible = self.solver.feasibility_solve(
            phase_one_tableau, num_variables, num_slack_variables
        )
        if not solution_is_feasible:
            print("Problem is infeasible.")
            return False, None
        tableau = self.remove_aux_variables(tableau, num_aux_variables)
        merged_values_values, values = self.read_solution(tableau, num_variables, slack_values)
        relu_violations = self.find_relu_violation_pairs(values)
        if 0 < len(relu_violations):
            attempt_count = [0 for _ in range(len(self.problem_contextualizer.relu_constraints))]
            solution_is_feasible, tableau, slack_values = self.relu_search(
                tableau, slack_values, num_variables, attempt_count
            )
        if solution_is_feasible:
            merged_values_values, values = self.read_solution(tableau, num_variables, slack_values)
            print("solution is feasible")
            print(f"values: {values}")
            print(f"merged values: {merged_values_values}")
            return True, merged_values_values
        print("solution is not feasible")
        return False, None
