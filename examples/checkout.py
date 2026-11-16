"""A shop's checkout: turn a cart into parcels.

Run it:

    PYTHONPATH=src python3 examples/checkout.py

This is the shape most projects start from. You already have product rows and cart lines
in your own format; the library wants a request. So there are two small functions between
them: `to_request` maps your records onto the request, and `parcels` reads the result
back into something your checkout page can render. Everything Packvium-specific lives in
those two functions, and the rest of the shop never sees a tick or a placement.

The request is a plain dict, the same JSON every Packvium engine reads, so the mapping
below works unchanged if you later move the solve to another language or a queue worker.
"""

from __future__ import annotations

from packvium import pack_from_dict

# Your product table. Sizes in centimetres and weights in grams, as a shop usually has them.
PRODUCTS = {
    "kettle": {"size_cm": (24, 22, 26), "weight_g": 1300, "this_side_up": True},
    "mug": {"size_cm": (12, 9, 10), "weight_g": 350, "this_side_up": False},
    "cookbook": {"size_cm": (28, 22, 3), "weight_g": 900, "this_side_up": False},
    "floor-lamp": {"size_cm": (150, 30, 30), "weight_g": 4200, "this_side_up": False},
}

# The cartons on your packing bench, with what each one costs you in cents.
BOXES = [
    {"id": "small", "inner_cm": (30, 25, 15), "max_kg": 5, "cost_cents": 45},
    {"id": "medium", "inner_cm": (40, 30, 30), "max_kg": 12, "cost_cents": 80},
    {"id": "large", "inner_cm": (60, 40, 40), "max_kg": 20, "cost_cents": 140},
]


def to_request(cart: dict[str, int]) -> dict:
    """Your cart (`{sku: quantity}`) as a Packvium request."""
    return {
        "units": {"length": "cm"},
        "configuration": {
            # Pay as little as possible for cartons, after packing everything that fits.
            "objective": "lowest_cost",
            # Counted work decides where the search stops, so the same cart always gets
            # the same parcels. The time limit is a fuse against a hang, not a search limit.
            "effort_budget": {"max_candidates_evaluated": 200_000},
            "time_limit_ms": 60_000,
        },
        "items": [
            {
                "id": sku,
                "quantity": quantity,
                "weight": f"{PRODUCTS[sku]['weight_g']} g",
                "keep_upright": PRODUCTS[sku]["this_side_up"],
                "dimensions": dict(zip(("length", "width", "height"), map(str, PRODUCTS[sku]["size_cm"]))),
            }
            for sku, quantity in cart.items()
        ],
        "containers": [
            {
                "id": box["id"],
                "max_payload": f"{box['max_kg']} kg",
                "cost_minor": box["cost_cents"],
                "inner_dimensions": dict(zip(("length", "width", "height"), map(str, box["inner_cm"]))),
            }
            for box in BOXES
        ],
    }


def parcels(cart: dict[str, int]) -> tuple[list[dict], list[str]]:
    """The parcels to ship, and the SKUs that fit no carton you stock."""
    result = pack_from_dict(to_request(cart))
    shipped = []
    for container in result["containers"]:
        contents: dict[str, int] = {}
        for placement in container["placements"]:
            contents[placement["item_type"]] = contents.get(placement["item_type"], 0) + 1
        shipped.append({
            "box": container["container_type"],
            "contents": contents,
            # `value` is the exact weight rendered in `unit`; here that is whole grams.
            "weight": f"{container['gross_weight']['value']} {container['gross_weight']['unit']}",
        })
    # An item that fits nowhere is not an error. It comes back with a reason, and the rest
    # of the cart is still packed.
    left_out = sorted({entry["item_type"] for entry in result["unpacked_items"]})
    return shipped, left_out


def show(title: str, cart: dict[str, int]) -> None:
    shipped, left_out = parcels(cart)
    print(f"\n{title}")
    for number, parcel in enumerate(shipped, start=1):
        contents = ", ".join(f"{count} x {sku}" for sku, count in sorted(parcel["contents"].items()))
        print(f"  parcel {number}: {parcel['box']:6s} {parcel['weight']:>7s}  {contents}")
    box_cost = sum(box["cost_cents"] for parcel in shipped for box in BOXES if box["id"] == parcel["box"])
    print(f"  cartons cost {box_cost} cents")
    for sku in left_out:
        print(f"  {sku} fits none of our cartons: ship it in its own packaging")


show("a small order", {"mug": 2, "cookbook": 1})
show("a bigger order", {"kettle": 4, "mug": 6, "cookbook": 4})
show("an order with something oversized", {"floor-lamp": 1, "mug": 4})
