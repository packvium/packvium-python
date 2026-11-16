"""From a paid order to labelled parcels: eligibility, packing, a second opinion, a price.

Run it:

    PYTHONPATH=src python3 examples/fulfilment.py

checkout.py maps a cart onto a request and reads the parcels back. A warehouse needs more
around that call, and this example wires the pieces in the order a fulfilment job runs them:

1. `evaluate_policy` decides which carrier service the order may use at all.
2. `pack_from_dict` packs it, under the rules the goods come with.
3. `validate` checks the plan against the request with code that shares nothing with the
   solver, so a parcel is never labelled on the solver's word alone.
4. `quote` prices every parcel from the tariff version you pinned.

Two kinds of trouble are kept apart on purpose. A request that is wrong (a quantity of
zero from a bad import) raises `InvalidRequestError`, and the job puts the order on hold.
An item that fits no carton is not an error: the rest of the order ships and the item is
handed to freight.
"""

from __future__ import annotations

from packvium import InvalidRequestError, pack_from_dict, validate
from packvium.commerce import evaluate_policy, quote

#: Results carry exact integers: 16 000 length ticks to the millimetre, 8 000 000 weight
#: ticks to the gram. A carrier bills whole millimetres and grams, rounded up.
TICKS_PER_MM = 16_000
TICKS_PER_GRAM = 8_000_000

# What you publish: one tariff per carrier service, and the rules that say what may ship
# how. You would load this from your own storage; a version's number is its position in
# the list, so `tariff_version: 1` below always means the same prices.
COMMERCE = {
    "tariffs": [
        {
            "carrier_id": "postway", "service_id": "air",
            "versions": [{
                "effective_at": 0,
                "dimensional_weight_divisor": 5000,
                "cost_per_dimensional_kg_minor": {"domestic": 520, "eu": 890},
                "minimum_charge_minor": 950,
                "fuel_surcharge_permille": 140,
                "accessorials": [],
            }],
        },
        {
            "carrier_id": "postway", "service_id": "ground",
            "versions": [{
                "effective_at": 0,
                "dimensional_weight_divisor": 6000,
                "cost_per_dimensional_kg_minor": {"domestic": 210, "eu": 380},
                "minimum_charge_minor": 590,
                "fuel_surcharge_permille": 90,
                "accessorials": [],
            }],
        },
    ],
    "policy_rules": [
        {
            "rule_id": "lithium-stays-on-the-ground",
            "versions": [{
                "scope": "service", "action": "reject", "priority": 10, "effective_at": 0,
                "reason": "lithium batteries do not fly with this carrier",
                "predicates": [
                    {"scope": "service", "field": "service", "operator": "equals", "value": "air"},
                    {"scope": "service", "field": "battery", "operator": "equals", "value": "lithium"},
                ],
            }],
        },
    ],
    "catalogs": [],
}

PRODUCTS = {
    "laptop": {"mm": (360, 250, 30), "g": 2100, "battery": "lithium"},
    "monitor": {"mm": (680, 430, 140), "g": 5400, "keep_upright": True, "stackable": False},
    "bleach": {"mm": (90, 90, 280), "g": 1100, "keep_upright": True,
               "tags": ["chemical"], "incompatible_tags": ["food"]},
    "coffee": {"mm": (120, 80, 220), "g": 1000, "tags": ["food"]},
    "desk": {"mm": (1600, 800, 60), "g": 28000},
}

BOXES = [
    {"id": "S", "mm": (300, 200, 300), "tare_g": 180, "max_kg": 8, "cost_minor": 50},
    {"id": "M", "mm": (400, 300, 300), "tare_g": 260, "max_kg": 15, "cost_minor": 85},
    {"id": "L", "mm": (700, 450, 450), "tare_g": 520, "max_kg": 25, "cost_minor": 160},
]


def measures(edges: tuple[int, int, int]) -> dict[str, str]:
    return dict(zip(("length", "width", "height"), map(str, edges)))


def allowed_service(order: dict) -> tuple[str, str | None]:
    """The service the customer chose, or ground when a rule forbids it, and why."""
    for sku in order["lines"]:
        decision = evaluate_policy(COMMERCE, {
            "scope": "service",
            "context": {"service": order["service"], "battery": PRODUCTS[sku].get("battery", "none")},
            "as_of": 0,
        })["decision"]
        if not decision["allowed"]:
            return "ground", f"{sku}: {decision['citation']['reason']}"
    return order["service"], None


def packing_request(order: dict) -> dict:
    items = []
    for sku, quantity in order["lines"].items():
        product = PRODUCTS[sku]
        item = {"id": sku, "quantity": quantity, "weight": f"{product['g']} g",
                "dimensions": measures(product["mm"])}
        # Only the rules a product actually has are sent; an absent field means "no rule".
        for rule in ("keep_upright", "stackable", "tags", "incompatible_tags"):
            if rule in product:
                item[rule] = product[rule]
        items.append(item)
    return {
        "units": {"length": "mm"},
        "configuration": {
            "objective": "lowest_cost",
            # Nothing may hang in the air: three quarters of every base must rest on something.
            "minimum_support_ratio": 0.75,
            "effort_budget": {"max_candidates_evaluated": 200_000},
            "time_limit_ms": 60_000,
        },
        "items": items,
        "containers": [
            {"id": box["id"], "inner_dimensions": measures(box["mm"]), "cost_minor": box["cost_minor"],
             "tare_weight": f"{box['tare_g']} g", "max_payload": f"{box['max_kg']} kg"}
            for box in BOXES
        ],
    }


def price(container: dict, service: str, zone: str) -> dict:
    """One parcel through the pinned tariff. Whole grams and millimetres, rounded up.

    The answer is either a quote or a rejection with a code: a zone the tariff does not
    price is never a zero.
    """
    grams = -(-container["gross_weight"]["ticks"] // TICKS_PER_GRAM)
    volume = 1
    for edge in ("length", "width", "height"):
        volume *= -(-container["outer_dimensions"][edge]["ticks"] // TICKS_PER_MM)
    return quote(COMMERCE, {
        "carrier_id": "postway", "service_id": service, "tariff_version": 1,
        "zone": zone, "actual_weight_g": grams, "volume_mm3": volume,
    })


def fulfil(order: dict) -> None:
    print(f"\norder {order['id']} to zone {order['zone']}, customer chose {order['service']}")
    service, why = allowed_service(order)
    if why:
        print(f"  moved to {service}: {why}")

    request = packing_request(order)
    try:
        result = pack_from_dict(request)
    except InvalidRequestError as error:
        # `field` is a JSON Pointer into the request, `reason` one of a closed set.
        print(f"  on hold: {error.reason} at {error.field} ({error.detail})")
        return

    issues = validate(request, result)
    if issues:
        print("  on hold: the plan failed its check: " + "; ".join(f"{i.code}: {i.detail}" for i in issues))
        return
    print(f"  plan checked: {len(result['containers'])} parcel(s), no issues")

    answers = [price(container, service, order["zone"]) for container in result["containers"]]
    refused = [answer["error"]["code"] for answer in answers if answer["status"] != "ok"]
    if refused:
        print(f"  on hold: no {service} price for zone {order['zone']} ({refused[0]})")
        return

    total = 0
    for container, answer in zip(result["containers"], answers):
        contents: dict[str, int] = {}
        for placement in container["placements"]:
            contents[placement["item_type"]] = contents.get(placement["item_type"], 0) + 1
        quoted = answer["quote"]
        total += quoted["total_minor"]
        listing = ", ".join(f"{count} x {sku}" for sku, count in sorted(contents.items()))
        print(f"  {container['id']:4s} {listing}")
        print(f"       {quoted['actual_weight_g']} g on the scale, billed as "
              f"{quoted['billed_weight_g']} g, {quoted['total_minor']} minor units")
    print(f"  shipping by {service}: {total} minor units")

    for entry in result["unpacked_items"]:
        print(f"  {entry['item_id']} goes to freight: {entry['reason']}")


fulfil({"id": "A-1001", "zone": "domestic", "service": "air",
        "lines": {"coffee": 6, "bleach": 2}})
fulfil({"id": "A-1002", "zone": "eu", "service": "air",
        "lines": {"laptop": 1, "monitor": 1, "desk": 1}})
fulfil({"id": "A-1003", "zone": "domestic", "service": "ground",
        "lines": {"coffee": 0}})
fulfil({"id": "A-1004", "zone": "overseas", "service": "ground",
        "lines": {"coffee": 1}})
