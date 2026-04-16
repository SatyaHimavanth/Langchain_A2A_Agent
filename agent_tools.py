from typing import TypeVar, Union
from decimal import Decimal
from fractions import Fraction

from langchain.tools import tool


Number = TypeVar("Number", int, float, complex, Decimal, Fraction)


@tool
def addition(a: Number, b: Number) -> Number:
    """Adds two numbers of any valid Python numeric type."""
    return a + b

@tool
def subtraction(a: Number, b: Number) -> Number:
    """Subtract the second number from the first."""
    return a - b

@tool
def multiplication(a: Number, b: Number) -> Number:
    """Multiply two numbers together."""
    return a * b

@tool
def division(a: Number, b: Number) -> Union[float, Decimal, Fraction]:
    """Divide the first number by the second. Returns a float/high-precision type."""
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero. Please provide a non-zero divisor.")
    return a / b


@tool
def power(base: Number, exponent: Number) -> Number:
    """Raise the base to the power of the exponent."""
    if base == 0 and exponent < 0:
        raise ValueError("0.0 cannot be raised to a negative power.")
    return pow(base, exponent)

@tool
def root(value: Number, index: Number = 2) -> Number:
    """
    Calculate the n-th root of a value. 
    Defaults to 2 (square root).
    """
    if index == 0:
        raise ZeroDivisionError("The root index cannot be zero.")
    if value == 0 and index < 0:
        raise ValueError("Cannot calculate a negative root of zero.")
    return pow(value, 1/index) # value ** (1 / index)


