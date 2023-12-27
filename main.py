"""Main file to launch reluplex program."""

import pandas as pd
from neural_network import BasicModel
from reluplex.reluplex import ReluPlex


def main():
    # initilize neural network we wish to analyze
    basic_model = BasicModel()
    basic_model.set_manual_weights([[[1], [-1]], [[]], [[1, 1]]])
    basic_model.set_manual_bias([[-0.4, 0], [], [0]])
    # initialize reluplex solver
    relu_plex = ReluPlex(basic_model.super_basic_model)
    # add constraints to input and output node
    relu_plex.add_lower_bound_constraint_to_input(0, 0.5)
    relu_plex.add_upper_bound_constraint_to_input(0, 1)
    relu_plex.add_lower_bound_constraint_to_output(0, 0.5)
    relu_plex.add_upper_bound_constraint_to_output(0, 2)
    # determine if there is a feasible point
    relu_plex.find_feasible_point()


if __name__ == "__main__":
    main()
