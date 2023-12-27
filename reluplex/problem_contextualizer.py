"""Problem contextualizer for Reluplex."""

from typing import List

import numpy as np
from reluplex.pydantic_dictionaries import (
    Constraint,
    ConstraintTypes,
    Operand,
    ReluConstraint,
    Variable,
)
from reluplex.tableau_generator import TableauFormatter
from torch import nn


class ProblemContextualizer:
    """Problem contextualizer for Reluplex."""

    def __init__(self, sequential_network: nn.Sequential):
        self.structured_variables = self._generate_neural_node_variables(sequential_network)
        self.problem_constraints, self.relu_constraints = self._find_implicit_constraints()
        self.not_non_negative_constraints = self.find_not_non_negative_constraints()
        self.tableau_formatter = TableauFormatter()

    def add_explicit_constraint(self, node_count: int, bound_to_set: float, input_node: bool, is_lower_bound: bool) -> None:
        """Add explicit constraint to problem."""
        # get variable that bound will be set to
        if input_node:
            target_variable = self.structured_variables[0][node_count]
        else:
            target_variable = self.structured_variables[-1][node_count]
        operand = Operand(variable=target_variable, weight=1)
        # add lower bound constraint
        if is_lower_bound:
            constraint = Constraint(
                right_hand_side=bound_to_set, constraint=ConstraintTypes.GREATER_THAN.value, operands=[operand]
            )
        #  add upper bound constraint
        else:
            constraint = Constraint(
                right_hand_side=bound_to_set, constraint=ConstraintTypes.LESS_THAN.value, operands=[operand]
            )
        self.problem_constraints.append(constraint)

    def _find_implicit_constraints(self) -> (List[Constraint], List[ReluConstraint]):
        """Find simplex constraints implicit in neural network structure."""
        constraints = []
        relu_constraints = []
        for layer_count, current_layer in enumerate(self.structured_variables[1:]):
            previous_layer = self.structured_variables[layer_count]
            for node_count, node_variable in enumerate(current_layer):
                if 0 < len(node_variable.associated_weights):
                    operands = []
                    operands.append(Operand(variable=node_variable, weight=-1.0))
                    for previous_variable, weight in zip(previous_layer, node_variable.associated_weights):
                        operands.append(Operand(variable=previous_variable, weight=weight))
                    constraints.append(
                        Constraint(
                            right_hand_side=-1 * node_variable.bias,
                            constraint=ConstraintTypes.EQUALITY.value,
                            operands=operands,
                        )
                    )
                else:
                    relu_constraints.append(
                        ReluConstraint(
                            input_node=previous_layer[node_count],
                            output_node=node_variable,
                            constraint_count=len(relu_constraints),
                        )
                    )
        return constraints, relu_constraints

    def _generate_neural_node_variables(self, sequential_network: nn.Sequential) -> List[List[Variable]]:
        """Sequential network transformation."""
        # Assign helper variables
        structured_variables = []
        layers_variables = []
        total_inclusive_variable_count = 0
        total_exclusive_variable_count = 0
        # create variables for each input node of the network
        for input_node_count in range(len(sequential_network[0].weight[0])):
            layers_variables.append(
                Variable(
                    layer=0,
                    position_in_layer=input_node_count,
                    position_in_flat_without_negatives=input_node_count,
                    positions_in_flat_array=[total_inclusive_variable_count, total_inclusive_variable_count + 1],
                    associated_weights=[],
                    bias=0,
                    is_non_negative=False,
                )
            )
            total_inclusive_variable_count += 2
            total_exclusive_variable_count += 1
        structured_variables.append(layers_variables)
        # iterate through all hidden layers and output layer
        for layer_count, layer in enumerate(sequential_network):
            layers_variables = []
            variable_count = 0
            # generate node for linear node
            if isinstance(layer, nn.Linear):
                weights = layer.weight.detach().numpy()
                bias = layer.bias.detach().numpy()
                for node_count, node_weights_set in enumerate(weights):
                    layers_variables.append(
                        Variable(
                            layer=layer_count + 1,
                            position_in_layer=variable_count,
                            position_in_flat_without_negatives=total_exclusive_variable_count,
                            positions_in_flat_array=[
                                total_inclusive_variable_count,
                                total_inclusive_variable_count + 1,
                            ],
                            associated_weights=list(node_weights_set),
                            bias=bias[node_count],
                            is_non_negative=False,
                        )
                    )
                    variable_count += 1
                    total_inclusive_variable_count += 2
                    total_exclusive_variable_count += 1
            # generate node for relu node
            else:
                for variable_count in range(len(structured_variables[-1])):
                    layers_variables.append(
                        Variable(
                            layer=layer_count + 1,
                            position_in_layer=variable_count,
                            positions_in_flat_array=[total_inclusive_variable_count],
                            position_in_flat_without_negatives=total_exclusive_variable_count,
                            associated_weights=[],
                            bias=0,
                            is_non_negative=True,
                        )
                    )
                    total_inclusive_variable_count += 1
                    total_exclusive_variable_count += 1
                    variable_count += 1
            structured_variables.append(layers_variables)
        return structured_variables

    def convert_constraints_to_tableau(self) -> (np.matrix, np.ndarray, List[ConstraintTypes]):
        """Convert problem constraints to cvxpy constraints."""
        total_variables = sum(len(layer) for layer in self.structured_variables)
        tableau = []
        right_hand_side_list = []
        constraint_type = []
        for linear_constraint in self.problem_constraints:
            new_row = np.zeros(total_variables)
            right_hand_side = linear_constraint.right_hand_side
            for operand in linear_constraint.operands:
                new_row[operand.variable.position_in_flat_without_negatives] = operand.weight
            if linear_constraint.constraint == ConstraintTypes.GREATER_THAN.value:
                new_row = -1 * new_row
                right_hand_side = -1 * right_hand_side
            tableau.append(new_row)
            right_hand_side_list.append(right_hand_side)
            constraint_type.append(linear_constraint.constraint)
        tableau = np.matrix(tableau)
        right_hand_side = np.array(right_hand_side_list)
        return tableau, right_hand_side, constraint_type

    def find_not_non_negative_constraints(self) -> List[int]:
        """Determine what neural node variables are not non-negative."""
        not_non_negative_constraints = []
        variable_count = 0
        for layer in self.structured_variables:
            for individual_variable in layer:
                if not individual_variable.is_non_negative:
                    not_non_negative_constraints.append(variable_count)
                variable_count += 1
        return not_non_negative_constraints

    def convert_to_standard_simplex_form(self) -> (np.matrix, np.ndarray, int, int, int):
        """Convert problem to tableau in standard simplex form."""
        tableau, right_hand_side, constraint_type = self.convert_constraints_to_tableau()
        return self.tableau_formatter.convert_problem_to_standard_form(
            tableau, right_hand_side, self.not_non_negative_constraints, constraint_type, self.relu_constraints
        )
