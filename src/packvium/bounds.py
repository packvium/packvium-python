"""Lower bounds on the objective vector, and the gap against an incumbent.

The mathematics is fixed by [docs/OPTIMALITY-CERTIFICATES.md](../../../docs/OPTIMALITY-CERTIFICATES.md)
and `scripts/optimality_bounds.py` is the independent oracle. This module is written from
the document and never imports the oracle, so the property tests compare two
implementations rather than one implementation with itself -- the discipline set
for irregular items and restated here.

What a bound is for. Every solver in this project is a heuristic: it returns an
arrangement and has no notion of what it did not try. A bound is the other half of that
sentence -- the best score the request *could* admit, computed by relaxing the problem
until it becomes arithmetic. Where the achieved score meets the bound, the search is over
whether or not it explored anything, and the engine knows it.

What it proves. These bounds relax geometry away entirely, and that is why attaining one is
a proof about the packing: every packing the request admits is a solution of the relaxation,
so none scores below its optimum, and a validated packing that attains the bound is optimal
among them. The proof covers only the rules this library models and only the result's own
objective; where the bound is missed, the gap is an upper limit on what a longer search
could gain, not a claim that the gain exists.

Instances that occupy less than their box. The document states this rule for
`nesting_height`: such an instance occupies less than its nominal volume, so nominal volumes
stop summing and every capacity argument built on them stops being a bound. The volume terms
are *dropped* rather than scaled -- a bound that is sometimes wrong is not a bound.

`convex_hull` and `compressible` have exactly the same property and the document does not
say so, because was written before the irregular shapes existed. A hull occupies
its hull rather than its bounding box, and a compressible item gives up height under load;
summing nominal box volumes over-states what a solution must carry, which over-states the
container count.
Measured on the golden corpus, that made the bound unsound on three fixtures --
`feature-convex-hull-wedges-share-one-crate`, `feature-route-hull-volume-reserve` and
`feature-compressible-volume-reserve-recomputes-loads` -- each claiming two containers were
necessary where one sufficed. They are treated here by the document's own rule.

That is the conservative repair, not the best one. A hull's exact volume is already computed
by `hull.shape_for`, and using it would keep the volume argument alive and tighter instead of
dropping it; a compressible item has a computable fully-compressed floor. Both are new
mathematics and belong to the design task, not to this implementation of it -- recorded in
`docs/OPTIMALITY-CERTIFICATES.md` rather than invented here.

Objective key order. `compute` returns the bounds in the *default* objective's order,
`(unpacked_count, container_count, total_cost_minor, unused_volume_ppm, stack_height_ppm)`.
`for_request` returns them in the order of the objective a result is scored with:
`lowest_cost` reads `(L0, L2, L1, L3, L4)`, which the document shows is still a bound;
`maximum_value`, `open_dimension_height` and `maximum_volume` each add one term for the key
only they rank by, and `shipping_cost` and `lowest_landed_cost` one each for what a
carrier bills, read from the configuration's dimensional-weight divisor.

Which instances. The bound is a property of the request, so it is taken over every instance
the request declares, fixed placements included. A bound over the free instances
alone counts a fixed item's volume as empty and over-states the unused-volume key.

Complexity. `O(n log n + c log c)` for `n` instances and `c` container types -- one sort of
the instance volumes, one of the weights, one of the container types by cost. No geometry is
touched and no candidate position is generated, which is why the packer computes it once per
pack, before any solver runs.
"""

from __future__ import annotations

import bisect
import functools
import heapq
from typing import TYPE_CHECKING, Dict, Iterable, Sequence, Tuple

from ._compat import dataclass
from .constraints import usable_volume
from .geometry import ShapeType, dimensional_weight
from .models import Container, Item, ItemInstance
from .units import Weight

if TYPE_CHECKING:
    from .config import PackingConfig

#: Parts per million, the fixed-point scale keys 3 and 4 of the objective are carried at.
PPM = 1_000_000

#: Every sum in the bound path must stay below this.
#:
#: Declared rather than inherited from the language. Python's integers are unbounded, PHP's
#: silently become doubles on overflow, JavaScript's `Number` stops being exact past 2^53 and
#: Rust's `i128` wraps -- so if each engine refused at its own limit the four would disagree
#: about which requests are answerable. The value comes from the widest intermediate the
#: formulas form: keys 3 and 4 multiply a summed volume by `PPM`, so `10^30 * 10^6 = 10^36`
#: sits about 170-fold inside `i128`. It never binds on a real request: `10^30` cubic ticks is
#: 244 million cubic metres, and a request that trips it has a data error in it.
MAX_BOUND_SUM = 10 ** 30

#: Largest integer every binding can return without changing its value.
#:
#: Intermediate arithmetic deliberately has the wider ``MAX_BOUND_SUM`` ceiling, but the
#: result crosses JSON and JavaScript's ``Number`` boundary.  Keeping the five public result
#: keys inside ``2**53 - 1`` makes "byte-identical" mean numerically identical as well as
#: textually similar; a larger cost must be refused before one engine rounds it.
MAX_BOUND_VALUE = 2 ** 53 - 1


#: The objectives a bound is stated for, and where each reads its bounds: the position in
#: `compute`'s default-order vector of each of the objective's own score keys, or the name of
#: the one key only that objective ranks by (docs/OPTIMALITY-GAP.md § 1.4).
OBJECTIVE_ORDERS: Dict[str, Tuple["int | str", ...]] = {
    "default": (0, 1, 2, 3, 4),
    "lowest_cost": (0, 2, 1, 3, 4),
    "maximum_value": (0, "value_forgone", 1, 2, 3),
    "open_dimension_height": (0, "achieved_height", 1, 2, 3),
    "maximum_volume": ("unpacked_volume_ppm", 0, 1, 2, 3),
    "shipping_cost": (0, "billed_weight", 1, 3, 4),
    "lowest_landed_cost": (0, "landed_charge", 1, 3, 4),
}

#: The objectives whose own key is what a carrier bills, which needs the configuration's
#: dimensional-weight divisor and units as well as the request.
BILLED_OBJECTIVES = ("shipping_cost", "lowest_landed_cost")


@dataclass(frozen=True, slots=True)
class Billing:
    """What a carrier can bill one container of a type, at the least.

    `tare` and `dimensional` are weight ticks: an opened container is billed
    `max(tare + payload, dimensional)`, so `max(tare, dimensional)` is its floor whatever it
    carries. `brackets` is its rate table as `(upper bound in weight ticks, charge in minor
    units)`, the charge already floored at the minimum and surcharged, or empty where the
    objective prices nothing.
    """

    tare: int
    dimensional: int
    brackets: Tuple[Tuple[int, int], ...] = ()

    @property
    def floor(self) -> int:
        return max(self.tare, self.dimensional)


class BoundOverflowError(ValueError):
    """A sum in the bound path exceeded the declared ceiling.

    A structured refusal rather than a number, because the alternative is the failure
    Baldacci et al. document for floating-point bin-packing solvers: a bound that is quietly
    wrong and carries no signal that it is. Every engine refuses at the same declared
    ceiling, so a request is either answerable everywhere or refused everywhere.
    """


class UnsoundBoundError(ValueError):
    """An achieved score fell below its own lower bound.

    One of the two is wrong, and saying so is more useful than reporting a negative gap.
    This is the assertion that fires first if a bound is ever made unsound, which is why it
    raises rather than clamping.
    """


@dataclass(frozen=True, slots=True)
class Bounds:
    """The five lower bounds, in the objective's own key order.

    Each is conditional on the keys before it: `container_count` is the least number of
    containers *given* that `unpacked_count` items were left behind, and so on down. That
    is what makes them comparable to a score vector key by key, and also why the gap stops
    at the first key that misses.
    """

    unpacked_count: int
    container_count: int
    total_cost_minor: int
    unused_volume_ppm: int
    stack_height_ppm: int

    def as_tuple(self) -> Tuple[int, int, int, int, int]:
        return (self.unpacked_count, self.container_count, self.total_cost_minor,
                self.unused_volume_ppm, self.stack_height_ppm)


@dataclass(frozen=True, slots=True)
class Gap:
    """How far an incumbent stands from the bound, on the first key that misses.

    `relative` is an exact rational pair `(numerator, denominator)` rather than a float:
    the reference arithmetic stays exact and each binding formats its own percentage
    without binary floating point becoming part of a certificate. It is `None` -- undefined,
    not substituted -- when the bound on that key is zero, which is the common case for
    `unpacked_count`. Dividing by a stand-in denominator would produce a number that looks
    like a percentage and is not one.
    """

    #: Index of the first key where the incumbent exceeds its bound, or `None` when every
    #: key is attained and the incumbent is optimal for this relaxation.
    key: "int | None"
    absolute: int
    relative: "Tuple[int, int] | None"

    @property
    def attained(self) -> bool:
        return self.key is None


def _capacity_total(values: Iterable["int | None"], quantities: Iterable["int | None"],
                    unbounded_when_value_infinite: bool) -> "int | None":
    """`Σc value · quantity`, or `None` for an unbounded total.

    `None` means infinity on the way in and on the way out. A limit nobody declared cannot
    be summed, and a single unlimited type makes the whole capacity unbounded -- except for
    volume, where a container with no usable volume adds nothing however many of it there
    are, which is why the caller says which rule applies.
    """
    total = 0
    for value, quantity in zip(values, quantities):
        if value is None:
            if unbounded_when_value_infinite:
                return None
            continue
        if quantity is None:
            if value > 0:
                return None
            continue
        total += value * quantity
    return _guard(total, "container capacity")


def _fit(ascending: Sequence[int], capacity: "int | None") -> int:
    """The largest `n` such that the `n` smallest values sum to at most `capacity`.

    Smallest first, and that is the whole soundness argument: taking the cheapest units
    maximises how many fit under one capacity, so this over-estimates what any real packing
    achieves. Geometry, support ratios, stacking rules, incompatible tags and route order
    can each make the real answer worse and none of them can make it better.

    The caller passes an already-ascending sequence. `compute` sorts the volumes and weights
    once for the later keys, and sorting them again here would have doubled the only
    superlinear work this module does.
    """
    if capacity is None:
        return len(ascending)
    used = 0
    for taken, cost in enumerate(ascending):
        used += cost
        if used > capacity:
            return taken
    return len(ascending)


def _guard(total: int, quantity: str) -> int:
    """Refuse a sum past the declared ceiling instead of carrying it further."""
    if total > MAX_BOUND_SUM:
        raise BoundOverflowError(
            f"{quantity} sums to {total}, above the {MAX_BOUND_SUM} ceiling the bound path "
            "declares; refusing rather than returning a number no engine can agree on"
        )
    return total


def _guard_output(value: int, quantity: str) -> int:
    """Refuse a result that cannot cross every binding exactly."""
    if value > MAX_BOUND_VALUE:
        raise BoundOverflowError(
            f"{quantity} bound is {value}, above the {MAX_BOUND_VALUE} exact portable "
            "result ceiling; refusing rather than rounding it in JavaScript"
        )
    return value


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _finite_max(values: Sequence["int | None"]) -> "int | None":
    """The largest declared limit, or `None` if any type declares none.

    One unlimited type makes the maximum unbounded and every term conditioned on it
    vacuous, which is why this collapses to `None` rather than ignoring the gap.
    """
    return None if any(value is None for value in values) else max(values)


def _occupies_less_than_its_box(item: Item) -> bool:
    """Can this item take up less room than its declared dimensions?

    Three ways, and every one of them breaks the same argument -- that nominal volumes sum
    to something a solution must carry. A nested item sinks into the one below it; a
    `convex_hull` occupies its hull and leaves the rest of its bounding box free, which is
    the entire reason that shape exists; a `compressible` item gives up height under load.

    The document names only the first because it predates the other two. Asking the question
    once, here, is what keeps a future fourth shape from reintroducing the same unsoundness
    silently -- a new `ShapeType` that is anything but a solid box has to pass this line.
    """
    if item.nesting_height is not None:
        return True
    return item.shape_type in (ShapeType.CONVEX_HULL, ShapeType.COMPRESSIBLE)


def for_request(instances: Sequence[ItemInstance], containers: Sequence[Container],
                objective: str, config: "PackingConfig | None" = None) -> "Tuple[int, ...] | None":
    """The request's bound in `objective`'s own score order, or `None` where none is stated.

    `instances` is every instance the request declares, fixed placements included, and
    `containers` every type it may open. `None` for an objective with no bound, for a billed objective whose inventory cannot
    carry a packing that attains `L0` at a price, for a request with a
    `made_to_measure` container (D13: the unused-volume key would be
    taken from the machine maximum while the score uses the fabricated box), and for a
    request past a declared ceiling. The last withholds the bound and nothing else: the pack
    is still answered, as it was before any bound existed (OPTIMALITY-GAP.md § 1.5).

    A function of the request alone, computed independently of the solver and deadline.
    Emitting its prefix also checks the result's score and the portable gap ceiling.
    """
    if objective not in OBJECTIVE_ORDERS:
        return None
    if any(container.made_to_measure is not None for container in containers):
        return None
    try:
        billings = billing_for(containers, config) if objective in BILLED_OBJECTIVES else ()
        return objective_vector(instances, containers, objective, billings)
    except (BoundOverflowError, BoundWithheld):
        return None


class BoundWithheld(ValueError):
    """A billed objective's inventory cannot carry, at any price it lists, a packing that
    attains `L0`; no answer is compared against a bound then, so none is stated."""


def billing_for(containers: Sequence[Container], config: "PackingConfig | None") -> Tuple[Billing, ...]:
    """Each type's `Billing`, from the objective's divisor and units.

    A rigid container is billed on its declared outer size, else its declared inner one; a
    pallet whose load overhangs is billed larger per side, never smaller, and dimensional
    weight only grows with each side, so the declared size is a floor. A bag is billed on the
    package its contents make, which can be smaller than declared, so its floor is 0.
    """
    if config is None or config.dimensional_weight_divisor is None:
        raise BoundWithheld("a billed objective needs configuration.dimensional_weight_divisor")
    landed = config.objective == "lowest_landed_cost"
    return tuple(Billing(
        tare=container.tare_weight.ticks,
        dimensional=0 if container.is_flat_bag else dimensional_weight(
            container.outer_dimensions or container.declared_inner,
            config.dimensional_weight_divisor, config.dimensional_weight_length_unit,
            config.dimensional_weight_weight_unit).ticks,
        brackets=_brackets(container) if landed else (),
    ) for container in containers)


def _brackets(container: Container) -> Tuple[Tuple[int, int], ...]:
    """The rate table as `(bound in weight ticks, charge)`; the charge at a bracket's bound is
    that bracket's charge, since the bound is the heaviest weight it prices."""
    table = container.rate_table
    if table is None:
        return ()
    return tuple((bound * Weight.TICKS_PER_G, table.charge_minor_or_none(bound))
                 for bound in table.weight_brackets_g)


def objective_vector(instances: Sequence[ItemInstance], containers: Sequence[Container],
                     objective: str, billings: Sequence[Billing] = ()) -> Tuple[int, ...]:
    """The bound in a covered `objective`'s score order, raising `BoundOverflowError` where a
    ceiling withholds it -- the predicate's or the objective's own term's -- and
    `BoundWithheld` where a billed objective has no priceable answer to bound."""
    vector = compute(instances, containers).as_tuple()
    return tuple(vector[position] if isinstance(position, int)
                 else _objective_term(position, instances, containers, vector, billings)
                 for position in OBJECTIVE_ORDERS[objective])


def _objective_term(name: str, instances: Sequence[ItemInstance], containers: Sequence[Container],
                    vector: Tuple[int, ...], billings: Sequence[Billing]) -> int:
    """The bound on the one key an objective ranks by and the default does not.

    Each is conditional on the keys before it, as `compute`'s are: the value and height terms
    hold for every packing that leaves exactly `unpacked` instances behind, and the volume term
    leads its objective and holds for every packing.

    No term sorts: `compute` already paid for the only sorts (finding F6 doubled that cost
    once), and each term needs at most the `unpacked` extreme values, which a heap selects in
    `O(n log L0)`. With `O(c)` over the container types, the request bound stays
    `O(n log n + c log c)`.
    """
    unpacked, opened = vector[0], vector[1]
    if name == "value_forgone":
        return _value_forgone_bound(instances, unpacked)
    if name in ("billed_weight", "landed_charge"):
        weights = sorted(instance.weight.ticks for instance in instances)
        quantities = [container.quantity for container in containers]
        if name == "billed_weight":
            return _billed_weight_bound(weights, billings, quantities, unpacked, opened)
        volumes = sorted(instance.item.dimensions.volume for instance in instances)
        nests = any(_occupies_less_than_its_box(instance.item) for instance in instances)
        limits = [(usable_volume(c), None if c.max_payload is None else c.max_payload.ticks,
                   c.max_items, c.quantity) for c in containers]
        return _landed_charge_bound(weights, volumes, nests, billings, limits, unpacked, opened)
    volumes = [instance.item.dimensions.volume for instance in instances]
    nests = any(_occupies_less_than_its_box(instance.item) for instance in instances)
    if name == "achieved_height":
        areas = [container.inner_dimensions.base_area for container in containers]
        return _achieved_height_bound(volumes, nests, unpacked, areas)
    usable = [usable_volume(container) for container in containers]
    return _unpacked_volume_bound(volumes, nests, usable, [c.quantity for c in containers])


def _smallest_from_inventory(values: Sequence[int], quantities: Sequence["int | None"],
                             count: int) -> "int | None":
    """The sum of the `count` smallest of `{value repeated quantity}`, or `None` when the
    inventory holds fewer. Sorting the types and taking `min(quantity, still needed)` of each
    is `L2`'s rule, `O(c log c)`, without building the multiset."""
    total, needed = 0, count
    for value, quantity in sorted(zip(values, quantities), key=lambda pair: pair[0]):
        if needed == 0:
            break
        taken = needed if quantity is None else min(quantity, needed)
        total += value * taken
        needed -= taken
    return None if needed else total


def _billed_weight_bound(weights: Sequence[int], billings: Sequence[Billing],
                         quantities: Sequence["int | None"], unpacked: int, opened: int) -> int:
    """`shipping_cost`'s second key, summed billed weight in ticks.

    A packing that leaves exactly `unpacked` behind opens at least `opened` containers and
    carries at least the lightest `n - unpacked` instances. Each opened container is billed
    `max(tare + payload, dimensional)`, at least its floor, and a sum of maxima is at least
    either sum on its own, so the two can be taken over different types. Every term is
    non-negative, so opening more only adds. `weights` is ascending.
    """
    if opened == 0:
        return 0
    floors = _smallest_from_inventory([b.floor for b in billings], quantities, opened)
    tares = _smallest_from_inventory([b.tare for b in billings], quantities, opened)
    if floors is None or tares is None:
        raise BoundWithheld("the inventory holds fewer containers than must be opened")
    payload = sum(weights[:len(weights) - unpacked])
    return _guard_output(max(floors, tares + payload), "billed weight")


def _landed_charge_bound(weights: Sequence[int], volumes: Sequence[int], nests: bool,
                         billings: Sequence[Billing],
                         limits: Sequence[Tuple[int, "int | None", "int | None", "int | None"]],
                         unpacked: int, opened: int) -> int:
    """`lowest_landed_cost`'s second key, the summed carrier charge in minor units.

    Every opened container is billed in exactly one bracket and pays its charge. A container
    in bracket `(bound, charge)` weighs at most `bound`, so it is reachable only when its floor
    is within it, carries at most `bound - tare` of payload (and its `max_payload`), and at
    most as many instances as the lightest-first count under that payload, its usable volume
    and `max_items`. Three relaxations of that assignment hold, and the largest is the bound:
    the cheapest reachable charge per container over `opened` containers; the least cost
    covering the placed weight; the least cost covering the placed count. Prices need not rise
    with weight, which is why a bracket is chosen rather than read off the floor. A type no
    bracket prices offers nothing: a packing that opens one is refused, never returned.
    `weights` and `volumes` are ascending; `limits` is `(usable, max_payload, max_items,
    quantity)` per type. `O(c b log n + c b log(c b))` for `b` brackets per type.
    """
    placed = len(weights) - unpacked
    if opened == 0 or placed == 0:
        return 0
    payload_prefix = _prefix_sums(weights)
    volume_prefix = None if nests else _prefix_sums(volumes)
    cheapest, quantities, offers = [], [], []
    for billing, (usable, max_payload, max_items, quantity) in zip(billings, limits):
        reachable = [(bound, charge) for bound, charge in billing.brackets
                     if charge is not None and billing.floor <= bound]
        if not reachable:
            continue
        cheapest.append(min(charge for _, charge in reachable))
        quantities.append(quantity)
        for bound, charge in reachable:
            carried = bound - billing.tare
            if max_payload is not None:
                carried = min(carried, max_payload)
            count = _fit_prefix(payload_prefix, carried)
            if volume_prefix is not None:
                count = min(count, _fit_prefix(volume_prefix, usable))
            if max_items is not None:
                count = min(count, max_items)
            offers.append((charge, carried, count, quantity))
    by_count = _smallest_from_inventory(cheapest, quantities, opened)
    by_weight = _cover([(charge, carried, quantity) for charge, carried, _, quantity in offers],
                       payload_prefix[placed - 1])
    by_items = _cover([(charge, count, quantity) for charge, _, count, quantity in offers], placed)
    if by_count is None or by_weight is None or by_items is None:
        raise BoundWithheld("no priceable inventory carries a packing that attains L0")
    return _guard_output(max(by_count, by_weight, by_items), "landed charge")


def _prefix_sums(ascending: Sequence[int]) -> list:
    totals, running = [], 0
    for value in ascending:
        running += value
        totals.append(running)
    return totals


def _fit_prefix(prefix: Sequence[int], capacity: int) -> int:
    """`_fit` by binary search over prefix sums, `O(log n)` per question."""
    return bisect.bisect_right(prefix, capacity)


def _cover(offers: Sequence[Tuple[int, int, "int | None"]], need: int) -> "int | None":
    """The least cost covering `need` units from `(cost, capacity, quantity)` offers, taken
    fractionally and rounded up, or `None` when the offers cannot cover it.

    A fractional covering problem with one constraint is solved exactly by taking offers in
    order of cost per unit. Only the last offer is taken in part, so the cost is an integer
    plus one ceiling and no fraction is carried. Offers of equal cost per unit give the same
    total in either order, so the order needs no tie-break. Each offer is limited by its
    type's whole inventory, which only weakens the bound.
    """
    if need <= 0:
        return 0
    ranked = sorted((offer for offer in offers if offer[1] > 0),
                    key=functools.cmp_to_key(lambda a, b: a[0] * b[1] - b[0] * a[1]))
    total = 0
    for cost, capacity, quantity in ranked:
        if quantity is None or quantity * capacity >= need:
            return total + _ceil_div(cost * need, capacity)
        total += cost * quantity
        need -= capacity * quantity
    return None


def _value_forgone_bound(instances: Sequence[ItemInstance], unpacked: int) -> int:
    """`maximum_value`'s second key: whichever `unpacked` instances are left behind, their
    value is at least that of the `unpacked` least valuable. Values are non-negative, so the
    running sum only grows, and it is refused as soon as it passes the portable ceiling."""
    total = 0
    for value in heapq.nsmallest(unpacked, (instance.item.value or 0 for instance in instances)):
        total = _guard_output(total + value, "value forgone")
    return total


def _achieved_height_bound(volumes, nests, unpacked, areas) -> int:
    """`open_dimension_height`'s second key, the summed tops in ticks.

    A container holding volume `v` on floor `A` stands at least `v / A` tall, because its
    placements are disjoint boxes inside `A × top`. Summed over the opened containers that is
    at least the placed volume over the widest floor, and the score is an integer. The placed
    volume is at least everything but the `unpacked` largest. An instance that occupies less
    than its box breaks the disjoint-boxes step, so the term is dropped.
    """
    floors = [area for area in areas if area > 0]
    if nests or unpacked >= len(volumes) or not floors:
        return 0
    placed = sum(volumes) - sum(heapq.nlargest(unpacked, volumes))
    return _guard_output(_ceil_div(placed, max(floors)), "achieved height")


def _unpacked_volume_bound(volumes, nests, usable, quantities) -> int:
    """`maximum_volume`'s first key: the volume no inventory can hold, as the score's own
    rounded-up share of the requested volume.

    The packed volume is at most the summed usable capacity, so at least the rest is left
    behind whatever the geometry does. Unlimited inventory holds everything, and an instance
    that occupies less than its box stops nominal volumes summing, so both give 0.
    """
    requested = sum(volumes)
    capacity = _capacity_total(usable, quantities, unbounded_when_value_infinite=False)
    if nests or requested == 0 or capacity is None:
        return 0
    return _ceil_div(max(0, requested - capacity) * PPM, requested)


def compute(instances: Sequence[ItemInstance], containers: Sequence[Container]) -> Bounds:
    """Every bound for one request, in the default objective's key order.

    The instances are every quantity-expanded instance the request declares -- free and
    fixed alike -- and the containers are the types it may open, both exactly as the request
    declares them, before any placement decision has been taken. Raises
    `BoundOverflowError` where the one refusal predicate of docs/OPTIMALITY-CERTIFICATES.md
    says the request is past a ceiling.
    """
    volumes = sorted(instance.item.dimensions.volume for instance in instances)
    weights = sorted(instance.weight.ticks for instance in instances)
    _guard(sum(volumes), "instance volume")
    _guard(sum(weights), "instance weight")
    nests = any(_occupies_less_than_its_box(instance.item) for instance in instances)
    count = len(instances)

    usable = [usable_volume(container) for container in containers]
    inner = [container.inner_dimensions.volume for container in containers]
    areas = [container.inner_dimensions.base_area for container in containers]
    heights = [container.inner_dimensions.height.ticks for container in containers]
    payloads = [None if c.max_payload is None else c.max_payload.ticks for c in containers]
    slots = [c.max_items for c in containers]
    quantities = [c.quantity for c in containers]
    costs = [c.cost_minor for c in containers]

    unpacked = _unpacked_bound(volumes, weights, nests, count, usable, payloads, slots,
                               quantities)
    placed = count - unpacked
    opened = _container_bound(volumes, weights, nests, placed, containers, usable, payloads,
                              slots)
    cost = _cost_bound(costs, quantities, opened)
    unused = _unused_volume_bound(volumes, nests, placed, inner, opened)
    height = _stack_height_bound(volumes, nests, placed, areas, heights, opened)
    return Bounds(
        _guard_output(unpacked, "unpacked count"),
        _guard_output(opened, "container count"),
        _guard_output(cost, "opening cost"),
        _guard_output(unused, "unused volume"),
        _guard_output(height, "stack height"),
    )


def _unpacked_bound(volumes, weights, nests, count, usable, payloads, slots,
                    quantities) -> int:
    """`L0`: remove the geometry entirely and ask what the declared resources alone forbid.

    Every instance becomes freely selectable, and only the three additive resources the
    request declares can stop one being placed. With two resources the minimum over each
    separately is a genuine relaxation and can fall below the true two-resource optimum; it
    remains a bound, which is all that is claimed.
    """
    volume_capacity = _capacity_total(usable, quantities, unbounded_when_value_infinite=False)
    payload_capacity = _capacity_total(payloads, quantities, unbounded_when_value_infinite=True)
    slot_capacity = _capacity_total(slots, quantities, unbounded_when_value_infinite=True)

    placeable = count
    if not nests:
        placeable = min(placeable, _fit(volumes, volume_capacity))
    placeable = min(placeable, _fit(weights, payload_capacity))
    if slot_capacity is not None:
        placeable = min(placeable, slot_capacity)
    return count - placeable


def _container_bound(volumes, weights, nests, placed, containers, usable, payloads,
                     slots) -> int:
    """`L1`: grant every container the largest capacity available, which can only understate
    how many are needed."""
    if placed <= 0 or not containers:
        return 0
    bound = 1
    if not nests:
        largest_usable = max(usable)
        if largest_usable > 0:
            bound = max(bound, _ceil_div(sum(volumes[:placed]), largest_usable))
    largest_payload = _finite_max(payloads)
    if largest_payload is not None and largest_payload > 0:
        bound = max(bound, _ceil_div(sum(weights[:placed]), largest_payload))
    largest_slots = _finite_max(slots)
    if largest_slots is not None and largest_slots > 0:
        bound = max(bound, _ceil_div(placed, largest_slots))
    return bound


def _cost_bound(costs, quantities, opened) -> int:
    """`L2`: at least `L1` containers open, each costing at least the cheapest the inventory
    still holds.

    Inventory is respected rather than assumed unlimited. Charging the cheapest type `L1`
    times would also be a bound, and a weaker one whenever that type is nearly exhausted.

    The types are sorted by cost once and taken greedily, `min(quantity, still needed)` of
    each, which is the sum of the `L1` cheapest entries of the per-unit multiset without
    building it: `O(c log c)` rather than the `O(c · L1 log(c · L1))` of materialising up to
    `L1` copies of every type and sorting those (finding F6).
    """
    needed = opened
    total = 0
    for cost, quantity in sorted(zip(costs, quantities), key=lambda pair: pair[0]):
        if needed <= 0:
            break
        taken = needed if quantity is None else min(quantity, needed)
        total += cost * taken
        needed -= taken
    return _guard(total, "opening cost")


def _unused_volume_bound(volumes, nests, placed, inner, opened) -> int:
    """`L3`: the fill is largest when every container is the smallest available and holds the
    greatest volume that could be placed at all.

    Key 3 sums a *per-container* ratio, so a tight bound would have to know which items went
    where -- that is the packing problem, not a relaxation of it. The `L1 - 1` term is the
    exact worst case by which summing `k` ceilings can exceed the ceiling of the sum.
    """
    if nests or opened <= 0 or not inner:
        return 0
    smallest_inner = min(inner)
    if smallest_inner <= 0:
        return 0
    largest_placed = sum(volumes[len(volumes) - placed:]) if placed > 0 else 0
    return max(0, opened * PPM - _ceil_div(largest_placed * PPM, smallest_inner) - (opened - 1))


def _stack_height_bound(volumes, nests, placed, areas, heights, opened) -> int:
    """`L4`: the volume that must be placed has to stand at least as tall as itself spread
    across the widest floor available, in the tallest container available.

    The `L1 - 1` correction is not a cautionary fudge. The objective floors each container's
    ratio *before* summing, and the sum of `k` floors can be one less than the floor of the
    sum at each of the first `k - 1` boundaries. Without it, two one-tick loads in height-6
    containers score `166666 + 166666 = 333332` while flooring their aggregate claims
    `333333` -- one ppm above a feasible solution, and therefore not a bound.
    """
    if nests or opened <= 0 or not areas:
        return 0
    widest = max(areas)
    tallest = max(heights)
    if widest <= 0 or tallest <= 0:
        return 0
    required = _ceil_div(sum(volumes[:placed]), widest) if placed > 0 else 0
    return max(0, required * PPM // tallest - (opened - 1))


def gap(score: Sequence[int], bound: Bounds) -> Gap:
    """The distance from an incumbent score to its bound, on the first key that misses.

    One key, not five: the keys after the first miss were computed on an assumption that has
    just been shown false -- that the earlier keys were attained -- so they say nothing and
    are not reported.
    """
    limits = bound.as_tuple()
    for index, (achieved, least) in enumerate(zip(score, limits)):
        if achieved < least:
            raise UnsoundBoundError(
                f"key {index} scored {achieved}, below its lower bound of {least}; "
                "one of the two is wrong"
            )
        if achieved > least:
            absolute = achieved - least
            return Gap(index, absolute, (absolute, least) if least > 0 else None)
    return Gap(None, 0, None)


def statement(score: Sequence[int], bound: Sequence[int]) -> "Dict[str, object] | None":
    """The `optimality` fields one result states against its request's bound (OPTIMALITY-GAP.md
    § 1.1), or `None` where § 1.5 withholds them.

    `bound` is in the result's own score order. The prefix runs through the first position the
    score misses, or is the whole bound when nothing is missed -- the attained case, which
    carries no gap at all rather than a zero one. A score below its bound means the bound is
    wrong, and an engine must not publish a claim it has just refuted, so nothing is stated.
    A gap past the portable result ceiling also withholds the whole statement.
    At most five comparisons, `O(1)`.
    """
    for index, (achieved, least) in enumerate(zip(score, bound)):
        if achieved < least:
            return None
        if achieved > least:
            absolute = achieved - least
            if absolute > MAX_BOUND_VALUE:
                return None
            fields: Dict[str, object] = {
                "lower_bound": list(bound[:index + 1]),
                "gap_key": index,
                "absolute_gap": absolute,
            }
            if least > 0:
                # Unreduced, so the pair reads as (score - bound, bound) by eye.
                fields["relative_gap"] = {"numerator": absolute, "denominator": least}
            return fields
    return {"lower_bound": list(bound)}
