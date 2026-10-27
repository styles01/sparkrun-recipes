"""
title: Calculator
author: tandem-tests
description: Arithmetic and unit conversions the model should not do in its head.
version: 0.2.0
license: MIT
"""

import math
from typing import Literal, Optional


class Tools:
    def __init__(self):
        pass

    def calculate(self, expression: str) -> str:
        """
        Evaluate an arithmetic expression and return the result.
        :param expression: The expression, for example "2 * (3 + 4) ** 2".
        """
        allowed = {k: getattr(math, k) for k in ("sqrt", "sin", "cos", "pi", "log")}
        return str(eval(expression, {"__builtins__": {}}, allowed))

    def convert_length(self, value: float, unit: Literal["km", "mi"], digits: Optional[int] = 2) -> str:
        """
        Convert a length between kilometres and miles.
        :param value: The length to convert.
        :param unit: The unit of the value.
        :param digits: Digits after the point.
        """
        out = value / 1.609344 if unit == "km" else value * 1.609344
        return f"{round(out, digits or 0)} {'mi' if unit == 'km' else 'km'}"

    def _helper(self):
        return None
