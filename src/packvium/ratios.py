from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Container

# Support ratios arrive as floats from the public API but must never decide feasibility
# in floating point. They are converted once to a scaled integer and every comparison
# below is exact. See docs/UNITS-AND-NUMERICS.md.
SUPPORT_SCALE = 1_000_000


def scaled_ratio(ratio: float) -> int:
    return int(ratio * SUPPORT_SCALE + 0.5)


def required_area(base_area: int, scaled: int) -> int:
    """Exact floor(base_area * scaled / SUPPORT_SCALE) without an overflowing product."""
    whole, remainder = divmod(base_area, SUPPORT_SCALE)
    return whole * scaled + (remainder * scaled) // SUPPORT_SCALE


def reserved_volume(container: Container) -> int:
    """Volume set aside for packing material, exact even for a metre-scale container.

    Not floating-point `ratio * volume`: that is exactly the double-precision loss
    this codebase's fixed-point design exists to avoid for a feasibility number.
    """
    return required_area(container.inner_dimensions.volume, scaled_ratio(container.void_fill_reserve_ratio))
