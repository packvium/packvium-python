from __future__ import annotations

from ._compat import dataclass
from ._request_error import refused, refused_type
from decimal import Decimal
from enum import Enum
from fractions import Fraction
import re
from typing import Any, Optional, Tuple, Type, TypeVar


#: The whitespace of `\s` over ASCII text (space, tab, newline, return, form feed, vertical tab).
_ASCII_SPACE = " \t\n\r\f\v"
_LENGTH_SUFFIX_RE = re.compile(
    r"(.+?)\s*(mm|cm|m|in|inch|inches|ft|ticks?)", re.IGNORECASE | re.ASCII
)
_WEIGHT_SUFFIX_RE = re.compile(
    r"(.+?)\s*(mg|kg|g|oz|lbs?|ticks?)", re.IGNORECASE | re.ASCII
)


def _strip_ascii(text: str) -> str:
    return text.strip(_ASCII_SPACE)


class Rounding(str, Enum):
    FLOOR = "floor"
    CEIL = "ceil"
    NEAREST = "nearest"


def _fraction(value: int | str | Decimal | Fraction) -> Fraction:
    if isinstance(value, bool):
        # `True` is an `int` to Python and `1` to arithmetic, but never a measure anyone wrote.
        raise refused_type("a boolean is not a measure")
    if isinstance(value, float):
        if not value.is_integer():
            raise refused_type("float measure input is intentionally rejected; use a decimal string")
        value = int(value)
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, Decimal):
        return Fraction(value)
    text = _strip_ascii(str(value)).replace("\u00a0", " ")
    negative = text.startswith("-")
    mixed = re.fullmatch(r"[+-]?([0-9]+)\s+([0-9]+)\s*/\s*([0-9]+)", text, re.ASCII)
    if mixed:
        whole, numerator, denominator = map(_digits, mixed.groups())
        # The sign comes from the written text, not from the value of the whole part, so
        # "-0 1/2" is minus one half rather than plus one half.
        magnitude = whole * denominator + numerator
        return Fraction(-magnitude if negative else magnitude, denominator)
    simple = re.fullmatch(r"[+-]?([0-9]+)\s*/\s*([0-9]+)", text, re.ASCII)
    if simple:
        numerator, denominator = map(_digits, simple.groups())
        return Fraction(-numerator if negative else numerator, denominator)
    # ASCII digits only: `\d` alone also matches fullwidth and Arabic-Indic digits, which the
    # other engines refuse. No exponent form (`1e3000000` would cost seconds before any range
    # check), no underscores, no bare ".5" or "5.".
    decimal = re.fullmatch(r"[+-]?([0-9]+)(?:\.([0-9]+))?", text, re.ASCII)
    if decimal is None:
        raise refused("wrong_type", f"invalid decimal or fraction: {text}")
    whole, places = decimal.group(1), decimal.group(2) or ""
    # A shared cap, not a Python limitation: PHP cannot hold 10**19 in an integer, and a
    # silently different accepted range between the two is exactly the kind of divergence
    # this library exists to avoid. Eighteen digits is far past what a tick can express.
    if len(places) > 18:
        raise refused("wrong_type", "decimal has more fractional digits than a tick can distinguish")
    scale = 10 ** len(places)
    magnitude = _digits(whole) * scale + int(places or "0")
    return Fraction(-magnitude if negative else magnitude, scale)


#: Past this many significant digits a measure is far above every admitted maximum (2^53 - 1
#: ticks is sixteen digits), so its exact value cannot change the verdict.
_SIGNIFICANT_DIGITS = 64


def _digits(text: str) -> int:
    """ASCII digits as an integer, saturated past `_SIGNIFICANT_DIGITS` significant digits.

    Reading the full number costs time quadratic in its length (a 1 MB weight took 48 s), and
    `int` refuses past 4300 digits, which counted leading zeros too. O(len(text)).
    """
    significant = text.lstrip("0")
    if len(significant) > _SIGNIFICANT_DIGITS:
        return 10 ** _SIGNIFICANT_DIGITS
    return int(significant or "0")


def _unit_key(unit: object) -> str | None:
    """A unit code as the tables key it, or None for a value that is no code at all: `None` or
    a number is an unknown unit, as the JSON path refuses it, not an AttributeError."""
    return _strip_ascii(unit).lower() if isinstance(unit, str) else None


def _non_negative(ticks: int, kind: str) -> int:
    """A measure read from the caller's input; `__post_init__` still guards the arithmetic,
    where a negative value is a defect rather than a caller's mistake."""
    if ticks < 0:
        raise refused("negative_measure", f"{kind} cannot be negative")
    return ticks


def _rounded(value: Fraction, rounding: Rounding) -> int:
    numerator, denominator = value.numerator, value.denominator
    quotient, remainder = divmod(abs(numerator), denominator)
    sign = -1 if numerator < 0 else 1
    if remainder == 0:
        return sign * quotient
    if rounding is Rounding.FLOOR:
        return -(quotient + 1) if sign < 0 else quotient
    if rounding is Rounding.CEIL:
        return -quotient if sign < 0 else quotient + 1
    doubled = remainder * 2
    if doubled < denominator:
        return sign * quotient
    if doubled > denominator:
        return sign * (quotient + 1)
    return sign * (quotient if quotient % 2 == 0 else quotient + 1)


#: The unit codes the request schema names wherever it asks for a unit by itself (an output
#: unit, a dimensional-weight unit). A measure's own unit may also be spelt out ("inch").
LENGTH_UNIT_CODES = ("mm", "cm", "m", "in", "ft")
WEIGHT_UNIT_CODES = ("mg", "g", "kg", "oz", "lb")


@dataclass(frozen=True, order=True, slots=True)
class Length:
    """Exact fixed-point length. One tick equals 1/16000 mm."""

    ticks: int
    TICKS_PER_MM = 16_000
    TICKS_PER_INCH = 406_400
    _MULTIPLIERS = {
        "tick": 1,
        "ticks": 1,
        "mm": TICKS_PER_MM,
        "millimeter": TICKS_PER_MM,
        "millimeters": TICKS_PER_MM,
        "cm": TICKS_PER_MM * 10,
        "m": TICKS_PER_MM * 1000,
        "in": TICKS_PER_INCH,
        "inch": TICKS_PER_INCH,
        "inches": TICKS_PER_INCH,
        "ft": TICKS_PER_INCH * 12,
    }

    def __post_init__(self) -> None:
        if self.ticks < 0:
            raise ValueError("length cannot be negative")

    @classmethod
    def of(cls, value: int | str | Decimal | Fraction, unit: str = "mm", rounding: Rounding = Rounding.NEAREST) -> "Length":
        key = _unit_key(unit)
        if key not in cls._MULTIPLIERS:
            raise refused("invalid_unit", f"unsupported length unit: {unit}")
        return cls(_non_negative(_rounded(_fraction(value) * cls._MULTIPLIERS[key], rounding), "length"))

    @classmethod
    def mm(cls, value: Any, rounding: Rounding = Rounding.NEAREST) -> "Length":
        return cls.of(value, "mm", rounding)

    @classmethod
    def inches(cls, value: Any, rounding: Rounding = Rounding.NEAREST) -> "Length":
        return cls.of(value, "in", rounding)

    @classmethod
    def parse(cls, value: Any, default_unit: str = "mm", rounding: Rounding = Rounding.NEAREST) -> "Length":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls.of(value["value"], value.get("unit", default_unit), rounding)
        if isinstance(value, float):
            # A fraction in binary floating point is not the decimal the caller wrote; an
            # integral float is exact, and JavaScript cannot send anything else for `100`.
            if not value.is_integer():
                raise refused_type("float length input is intentionally rejected; use a decimal string")
            value = int(value)
        if isinstance(value, str):
            match = _LENGTH_SUFFIX_RE.fullmatch(_strip_ascii(value))
            if match:
                return cls.of(_strip_ascii(match.group(1)), match.group(2), rounding)
        return cls.of(value, default_unit, rounding)

    def as_fraction(self, unit: str = "mm") -> Fraction:
        key = unit.lower()
        if key not in self._MULTIPLIERS:
            raise ValueError(f"unsupported length unit: {unit}")
        return Fraction(self.ticks, self._MULTIPLIERS[key])

    def decimal(self, unit: str = "mm", places: int = 8) -> str:
        value = Decimal(self.as_fraction(unit).numerator) / Decimal(self.as_fraction(unit).denominator)
        return format(value.quantize(Decimal(1).scaleb(-places)).normalize(), "f")

    def to_dict(self, unit: str = "mm") -> dict[str, int | str]:
        return {"ticks": self.ticks, "value": self.decimal(unit), "unit": unit}


@dataclass(frozen=True, order=True, slots=True)
class Weight:
    """Exact weight. One tick equals 1/8 microgram."""

    ticks: int
    TICKS_PER_MG = 8_000
    TICKS_PER_G = 8_000_000
    TICKS_PER_KG = 8_000_000_000
    TICKS_PER_OZ = 226_796_185
    TICKS_PER_LB = 3_628_738_960
    _MULTIPLIERS = {
        "tick": 1,
        "ticks": 1,
        "mg": TICKS_PER_MG,
        "g": TICKS_PER_G,
        "kg": TICKS_PER_KG,
        "oz": TICKS_PER_OZ,
        "lb": TICKS_PER_LB,
        "lbs": TICKS_PER_LB,
    }

    def __post_init__(self) -> None:
        if self.ticks < 0:
            raise ValueError("weight cannot be negative")

    @classmethod
    def of(cls, value: Any, unit: str = "g", rounding: Rounding = Rounding.NEAREST) -> "Weight":
        key = _unit_key(unit)
        if key not in cls._MULTIPLIERS:
            raise refused("invalid_unit", f"unsupported weight unit: {unit}")
        return cls(_non_negative(_rounded(_fraction(value) * cls._MULTIPLIERS[key], rounding), "weight"))

    @classmethod
    def parse(cls, value: Any, default_unit: str = "g", rounding: Rounding = Rounding.NEAREST) -> "Weight":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls.of(value["value"], value.get("unit", default_unit), rounding)
        if isinstance(value, float):
            # A fraction in binary floating point is not the decimal the caller wrote; an
            # integral float is exact, and JavaScript cannot send anything else for `100`.
            if not value.is_integer():
                raise refused_type("float weight input is intentionally rejected; use a decimal string")
            value = int(value)
        if isinstance(value, str):
            match = _WEIGHT_SUFFIX_RE.fullmatch(_strip_ascii(value))
            if match:
                return cls.of(_strip_ascii(match.group(1)), match.group(2), rounding)
        return cls.of(value, default_unit, rounding)

    def decimal(self, unit: str = "g", places: int = 8) -> str:
        multiplier = self._MULTIPLIERS[unit.lower()]
        value = Decimal(self.ticks) / Decimal(multiplier)
        return format(value.quantize(Decimal(1).scaleb(-places)).normalize(), "f")

    def to_dict(self, unit: str = "g") -> dict[str, int | str]:
        return {"ticks": self.ticks, "value": self.decimal(unit), "unit": unit}


_Measure = TypeVar("_Measure", Length, Weight)


class ParsedMeasures:
    """The measures one request spells, each spelling parsed once.

    The request check parses every measure to refuse a bad one, and deserialisation reads the
    same spellings again to build the model. A parse is a pure function of the spelling, its
    kind and the default unit, so a repeated read returns the first one's value. The key carries
    the spelling's type, so `True` never answers for `1`; a `{value, unit}` object is keyed by
    the two fields `parse` reads, each with its type, so `{"value": 1}` never answers for
    `{"value": "1"}`. A refused spelling is not kept, so it is refused again with the same
    error, and an object with a field the key cannot hash is parsed on every read. Lives for one request: O(1) per
    read, O(distinct spellings) memory."""

    __slots__ = ("_parsed",)

    def __init__(self) -> None:
        self._parsed: dict[tuple[Any, ...], Any] = {}

    def read(self, kind: Type[_Measure], value: Any, default_unit: str) -> _Measure:
        if not isinstance(value, (str, int, float)):
            return self._read_object(kind, value, default_unit)
        key = (kind, type(value), value, default_unit)
        parsed = self._parsed.get(key)
        if parsed is None:
            parsed = self._parsed[key] = kind.parse(value, default_unit)
        return parsed

    def _read_object(self, kind: Type[_Measure], value: Any, default_unit: str) -> _Measure:
        key = _object_key(kind, value, default_unit)
        if key is None:
            return kind.parse(value, default_unit)
        try:
            parsed = self._parsed.get(key)
        except TypeError:
            # An unhashable field (a list as `value`) is the parser's to refuse, as it was
            # before objects were kept.
            return kind.parse(value, default_unit)
        if parsed is None:
            parsed = self._parsed[key] = kind.parse(value, default_unit)
        return parsed


def _object_key(kind: type, value: Any, default_unit: str) -> Optional[Tuple[Any, ...]]:
    """The two fields `kind.parse` reads from a `{value, unit}` object, each with its type, or
    None for anything else, which is parsed on every read. An object without `value` is left
    to `parse` to refuse."""
    if not isinstance(value, dict) or "value" not in value:
        return None
    field, unit = value["value"], value.get("unit", default_unit)
    return (kind, dict, type(field), field, type(unit), unit)


def output_length_unit(request_unit: str) -> str:
    """The output unit a request without `output.length_unit` is written in.

    The request's own `units.length`, spelt as the schema's output code: case and surrounding
    space are ignored and a spelt-out name folds to its code (`"Inches"` is `in`). A unit no
    output code names -- the internal `tick` -- gives `mm`, the default of both blocks."""
    multiplier = Length._MULTIPLIERS.get(_unit_key(request_unit))
    return next((code for code in LENGTH_UNIT_CODES if Length._MULTIPLIERS[code] == multiplier), "mm")
