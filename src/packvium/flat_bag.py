"""The flat-bag girth rule (docs/DEFORMABLE-CONTAINERS.md "1.6 readiness and contract").

A `flat_bag` container is two sheets of usable flat size `L x W` filled to at most thickness
`H`, its `inner_dimensions`. With `a`, `b` and `c` the largest envelope `x2`, `y2` and `z2`
over its placements, measured from the container origin, a packing is admitted only when the
ordinary boundary holds and `a + c <= L` and `b + c <= W`: a loop of film round the package's
cross-section is at least as long as that section's perimeter. Exact integer additions and
comparisons, so every check here is `O(1)` and needs no wider type than a coordinate.

The extents only grow as boxes are added, so a verdict reached for a candidate never flips
when more items follow, and any subset of an admitted set is admitted.

A `gusseted_bag` (1.10) has side gussets of depth `G` folded into each long edge: the tube
round the package's `y-z` section is `2(W + G)` long instead of `2W`, so the width sum becomes
`b + c <= W + G`, and `G = 0` is the flat bag.

A bottom gusset (1.11) is one panel of depth `K` folded into the closed end: the loop
along the bag runs front, bottom, back, `2L + K`, so the length sum becomes
`2(a + c) <= 2L + K`, doubled to stay in integers. A gusseted bag with a bottom gusset may
stand on it (`orientation: standing`): the solving frame is then `W x H x L`, the floor is the
opened bottom, and the same two loops go round other axes -- `a + b <= W + G` across,
`2(b + c) <= 2L + K` along. `Girth` holds the two loops for either frame.
"""

from __future__ import annotations

from typing import Iterable

from ._compat import dataclass
from .geometry import Dimensions
from .units import Length

FLAT_BAG = "flat_bag"
GUSSETED_BAG = "gusseted_bag"
DEFORMABLE_BOUNDARY_KINDS = (FLAT_BAG, GUSSETED_BAG)

FLAT = "flat"
STANDING = "standing"
BAG_ORIENTATIONS = (FLAT, STANDING)

Extent = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class Girth:
    """The two loops of film round a bag's package, as axis pairs of the solving frame.

    `along` is the loop down the length (`2L + K`, compared against the doubled sum of its two
    extents), `across` the loop round the width (`W + G`, against their plain sum). Lying flat
    they are `(x, z)` and `(y, z)`; standing, `(y, z)` and `(x, y)`.
    """

    along: tuple[int, int]
    along_limit: int
    across: tuple[int, int]
    across_limit: int

    def admits(self, extent: Extent) -> bool:
        """Whether a package of extents `(a, b, c)` passes both loops. The rigid boundary is
        the caller's own check and stays where it already is; only the two sums are new."""
        i, j = self.along
        p, q = self.across
        return (2 * (extent[i] + extent[j]) <= self.along_limit
                and extent[p] + extent[q] <= self.across_limit)


def girth_for(length: int, width: int, side_gusset: int, bottom_gusset: int, standing: bool) -> Girth:
    """The loops of a bag declared `length x width` with the two gusset depths, `0` where none
    is declared, in its flat or standing frame."""
    if standing:
        return Girth((1, 2), 2 * length + bottom_gusset, (0, 1), width + side_gusset)
    return Girth((0, 2), 2 * length + bottom_gusset, (1, 2), width + side_gusset)


def standing_dimensions(declared: Dimensions) -> Dimensions:
    """The `W x H x L` frame a bag declared `L x W x H` stands in."""
    return Dimensions(declared.width, declared.height, declared.length)


def grown_extent(extent: Extent, x2: int, y2: int, z2: int) -> Extent:
    """The running extents once a box reaching `(x2, y2, z2)` is added."""
    return (max(extent[0], x2), max(extent[1], y2), max(extent[2], z2))


def extent_of(boxes: Iterable[tuple[int, int, int]]) -> Extent:
    """`(max x2, max y2, max z2)` over upper corners, `(0, 0, 0)` for none. O(p)."""
    extent = (0, 0, 0)
    for x2, y2, z2 in boxes:
        extent = grown_extent(extent, x2, y2, z2)
    return extent


def girth_admits(inner: Dimensions, a: int, b: int, c: int, gusset: int = 0, bottom: int = 0) -> bool:
    """Whether a flat-lying bag declared `inner` admits a package of extents `(a, b, c)`, with a
    side gusset `gusset` and a bottom gusset `bottom` in ticks. The 1.10 signature, widened by a
    trailing `bottom`; the rigid boundary stays the caller's own check."""
    return girth_for(inner.length.ticks, inner.width.ticks, gusset, bottom, False).admits((a, b, c))


def fits_alone(inner: Dimensions, dimensions: Dimensions, girth: "Girth | int" = 0) -> bool:
    """Whether one box of these dimensions, alone at the origin of the frame `inner`, fits a bag.

    `girth` is the bag's two loops; a plain integer is the 1.10 form, a flat-lying bag's side
    gusset depth in ticks."""
    if isinstance(girth, int):
        girth = girth_for(inner.length.ticks, inner.width.ticks, girth, 0, False)
    extent = (dimensions.length.ticks, dimensions.width.ticks, dimensions.height.ticks)
    return dimensions.fits_inside(inner) and girth.admits(extent)


def as_dimensions(extent: Extent) -> Dimensions:
    """The achieved extent as dimensions; a packed bag always holds at least one box."""
    return Dimensions(Length(extent[0]), Length(extent[1]), Length(extent[2]))
