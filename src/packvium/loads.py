from __future__ import annotations

from typing import Sequence

from ._compat import dataclass
from .geometry import AxisAlignedBox
from .models import Placement


@dataclass(frozen=True, slots=True)
class LoadUnit:
    """The only facts load propagation needs about a box, real or hypothetical."""
    box: AxisAlignedBox
    weight_ticks: int
    max_top_load_ticks: int | None
    max_stacked_items: int | None
    label: str
    nesting_item_id: str | None = None
    nesting_height_ticks: int | None = None
    # Set only for a `compressible` item. Load propagation already computes the
    # cumulative mass above every unit, which is exactly the numerator the pressure model
    # needs, so the crush check rides the graph that is built anyway rather than a second one.
    compression_ratio_ppm: int | None = None
    max_compression_pressure_kpa: int | None = None
    # The item's own footprint, which the load presses on; `box` is its envelope, grown by the
    # clearance, and it carries the support graph.
    footprint_ticks2: int | None = None


def load_units(placements: Sequence[Placement], extra: LoadUnit | None = None) -> tuple[LoadUnit, ...]:
    units = [
        LoadUnit(p.envelope_box, p.instance.weight.ticks,
                 None if p.instance.item.max_top_load is None else p.instance.item.max_top_load.ticks,
                 p.instance.item.max_stacked_items,
                 p.instance.id,
                 None if p.instance.item.nesting_height is None else p.instance.item.id,
                 None if p.instance.item.nesting_height is None else p.instance.item.nesting_height.ticks,
                 p.instance.item.compression_ratio_ppm,
                 p.instance.item.max_compression_pressure_kpa,
                 p.dimensions.length.ticks * p.dimensions.width.ticks)
        for p in placements
    ]
    if extra is not None: units.append(extra)
    return tuple(units)
