"""Pydantic dictionaries for Reluplex."""
import uuid
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ConstraintTypes(Enum):
    """Available constraint types."""

    EQUALITY = "equality"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"


class Variable(BaseModel):
    """Object to contain information about a neural node variable."""

    id: str = Field(default=uuid.uuid4())
    layer: int
    position_in_layer: int
    positions_in_flat_array: List[int]
    position_in_flat_without_negatives: int
    associated_weights: List[float]
    bias: float
    is_non_negative: bool = True


class Operand(BaseModel):
    """Object to contain information of an operand operation."""

    variable: Variable
    weight: float


class Constraint(BaseModel):
    """Object to contain information needed for constraint."""

    right_hand_side: float
    constraint: ConstraintTypes
    operands: List[Operand]
    has_flex_variable: Optional[bool] = False
    has_artificial_variable: Optional[bool] = False

    class Config:
        """Config to allow for enum value type hinting."""

        use_enum_values = True


class ReluConstraint(BaseModel):
    """Object to contain information needed for relu constraint."""

    input_node: Variable
    output_node: Variable
    constraint_count: int
