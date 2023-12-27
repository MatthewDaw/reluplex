"""Pytorch neural network definitions."""
from typing import List
import torch
from torch import nn


class BasicModel(nn.Module):
    """Basic model for testing."""

    def __init__(self):
        super().__init__()

        self.super_basic_model = nn.Sequential(nn.Linear(1, 2, bias=True), nn.ReLU(), nn.Linear(2, 1, bias=True))

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        input_tensor = self.super_basic_model(input_tensor)
        return input_tensor

    def set_manual_weights(self, weight_value_list: List[List[List[float]]]) -> None:
        """Manually assign weights to network."""
        with torch.no_grad():
            for layer_count in range(len(weight_value_list)):
                if isinstance(self.super_basic_model[layer_count], nn.Linear):
                    self.super_basic_model[layer_count].weight = torch.nn.parameter.Parameter(
                        torch.tensor(weight_value_list[layer_count]).float()
                    )

    def set_manual_bias(self, bias_values: List[List[float]]) -> None:
        """Manually assign bias to network."""
        with torch.no_grad():
            for layer_count in range(len(bias_values)):
                if isinstance(self.super_basic_model[layer_count], nn.Linear):
                    self.super_basic_model[layer_count].bias = torch.nn.parameter.Parameter(
                        torch.tensor(bias_values[layer_count]).float()
                    )
