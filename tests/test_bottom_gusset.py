"""1.11: bottom gussets, lying flat or standing, admitted by the two loops of film."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

from packvium import validate
from packvium.flat_bag import girth_for
from packvium.geometry import Dimensions
from packvium.models import Container
from packvium.pallet_overhang import solving_frame
from packvium.request_errors import InvalidRequestError
from packvium.serialization import pack_from_dict
from packvium.units import Length

#: The independent oracle lives in the workspace's `scripts/`, not in the published package.
ORACLE = Path(__file__).resolve().parents[2] / "scripts" / "deformable_boundary_model.py"


def ticks(millimetres: int) -> int:
    return Length.of(millimetres, "mm").ticks


def request(boundary: dict, item: dict, quantity: int, **container) -> dict:
    return {"items": [{"id": item["id"], "quantity": quantity, "weight": "300 g", **item}],
            "containers": [{"id": "pouch", "quantity": 1, "deformable_boundary": boundary,
                            "inner_dimensions": {"length": "300", "width": "150", "height": "100"},
                            **container}],
            "configuration": {"solver_profile": "fast", "time_limit_ms": 60000,
                              "effort_budget": {"max_search_nodes": 100000}}}


ROD = {"id": "rod", "allowed_rotations": ["LWH"],
       "dimensions": {"length": "300", "width": "100", "height": "20"}}
BOTTLE = {"id": "bottle", "allowed_rotations": ["LWH"],
          "dimensions": {"length": "50", "width": "50", "height": "150"}}


@pytest.mark.skipif(not ORACLE.is_file(), reason="the independent oracle lives in the workspace only")
@pytest.mark.parametrize("seed", range(60))
def test_the_engine_loops_agree_with_the_oracle_on_every_extent(seed):
    """The engine's `Girth` and the independent oracle decide every package alike, flat or
    standing, over the loops alone (the rigid frame is each one's own separate check)."""
    sys.path.insert(0, str(ORACLE.parent))
    from deformable_boundary_model import FlatBag, girth_violation
    rng = random.Random(2604 + seed)
    length, width, thickness = rng.randint(2, 40), rng.randint(2, 40), rng.randint(1, 40)
    side, bottom = rng.randint(0, width), rng.randint(1, 2 * length)
    standing = seed % 2 == 1
    girth = girth_for(length, width, side, bottom, standing)
    bag = FlatBag(length, width, thickness, side_gusset_depth=side, bottom_gusset_depth=bottom,
                  standing=standing)
    frame = (width, thickness, length) if standing else (length, width, thickness)
    for _ in range(80):
        extent = tuple(rng.randint(0, side_limit) for side_limit in frame)
        assert girth.admits(extent) == (girth_violation(bag, extent) is None), (extent, standing)


def test_a_bottom_gusset_admits_a_rod_the_flat_length_refuses():
    """A 300 mm rod 20 mm thick fills the flat length, so a + c = 320 > 300 lying flat; a
    40 mm bottom gusset makes the loop 2L + K = 640 = 2(a + c), and equality fits."""
    flat = pack_from_dict(request({"kind": "gusseted_bag", "side_gusset_depth": "10"}, ROD, 1))
    assert flat["unpacked_items"][0]["reason"] == "no_compatible_container_dimensions"
    block = pack_from_dict(request({"kind": "gusseted_bag", "bottom_gusset_depth": "40"}, ROD, 1))
    assert block["unpacked_items"] == []
    past = pack_from_dict(request({"kind": "gusseted_bag",
                                   "bottom_gusset_depth": {"value": str(ticks(40) - 1), "unit": "ticks"}},
                                  ROD, 1))
    assert past["unpacked_items"][0]["reason"] == "no_compatible_container_dimensions"


def test_a_standing_pouch_packs_in_its_turned_frame_and_the_validator_agrees():
    """Standing, the 300 x 150 x 100 pouch is a 150 x 100 x 300 frame. Two 50 x 50 x 150
    bottles stand side by side, here front to back (a + b = 150 = W + G, b <= H); a third
    beside them breaks the loop round the section and one on top breaks the loop down the
    bag, 2(b + c) = 700 > 660."""
    standing = request({"kind": "gusseted_bag", "bottom_gusset_depth": "60",
                        "orientation": "standing"}, BOTTLE, 3)
    result = pack_from_dict(standing)
    packed = result["containers"][0]
    assert len(packed["placements"]) == 2 and len(result["unpacked_items"]) == 1
    assert {side: packed["achieved_dimensions"][side]["value"]
            for side in ("length", "width", "height")} == {"length": "50", "width": "100", "height": "150"}
    assert {side: packed["inner_dimensions"][side]["value"]
            for side in ("length", "width", "height")} == {"length": "300", "width": "150", "height": "100"}
    assert validate(standing, result) == []


def test_the_validator_refuses_a_standing_package_past_its_loop():
    """The same two bottles against a pouch 1 mm narrower: a + b = 150 > W = 149."""
    standing = request({"kind": "gusseted_bag", "bottom_gusset_depth": "60",
                        "orientation": "standing"}, BOTTLE, 2)
    result = pack_from_dict(standing)
    narrower = {**standing, "containers": [{**standing["containers"][0], "deformable_boundary": {
        "kind": "gusseted_bag", "orientation": "standing",
        "bottom_gusset_depth": "60"}}]}
    narrower["containers"][0]["inner_dimensions"] = {"length": "300", "width": "149", "height": "100"}
    codes = [issue.code for issue in validate(narrower, result)]
    assert "bag_girth_exceeded" in codes


def test_the_standing_frame_turns_once_and_declares_the_bag_as_given():
    declared = Dimensions.mm(300, 150, 100)
    bag = Container("pouch", declared, deformable_boundary="gusseted_bag",
                    bottom_gusset_depth=Length(ticks(60)), bag_orientation="standing")
    frame = solving_frame(bag)
    assert frame.inner_dimensions == Dimensions.mm(150, 100, 300)
    assert frame.declared_inner == declared
    assert solving_frame(frame) is frame
    assert frame.girth == girth_for(ticks(300), ticks(150), 0, ticks(60), True)


@pytest.mark.parametrize(("fields", "reason", "detail"), [
    ({}, "missing_field", "a gusseted_bag requires side_gusset_depth"),
    ({"bottom_gusset_depth": Length(ticks(600) + 1)}, "invalid_value",
     "bottom_gusset_depth exceeds twice inner_dimensions length"),
    ({"bottom_gusset_depth": Length(0)}, "invalid_value", "bottom_gusset_depth must be positive"),
    ({"side_gusset_depth": Length(ticks(10)), "bag_orientation": "standing"}, "missing_field",
     "bottom_gusset_depth is required when orientation is standing"),
    ({"bottom_gusset_depth": Length(ticks(10)), "bag_orientation": "upright"}, "not_allowed",
     "orientation must be one of ['flat', 'standing']"),
])
def test_the_object_api_refuses_what_the_json_path_refuses(fields, reason, detail):
    with pytest.raises(Exception) as caught:
        Container("pouch", Dimensions.mm(300, 150, 100), deformable_boundary="gusseted_bag", **fields)
    assert getattr(caught.value, "reason", None) == reason and detail in str(caught.value)


@pytest.mark.parametrize("field", ["bottom_gusset_depth", "bag_orientation"])
def test_a_rigid_box_or_flat_bag_refuses_the_gusseted_fields(field):
    value = "standing" if field == "bag_orientation" else Length(ticks(10))
    for boundary in (None, "flat_bag"):
        with pytest.raises(Exception):
            Container("box", Dimensions.mm(300, 150, 100), deformable_boundary=boundary, **{field: value})


def test_the_json_path_refuses_a_standing_bag_without_a_bottom_gusset():
    with pytest.raises(InvalidRequestError) as caught:
        pack_from_dict(request({"kind": "gusseted_bag", "side_gusset_depth": "10",
                                "orientation": "standing"}, BOTTLE, 1))
    assert (caught.value.reason, caught.value.field) == (
        "missing_field", "/containers/0/deformable_boundary/bottom_gusset_depth")


def test_the_1_10_helpers_keep_their_signatures():
    """`girth_admits` and `fits_alone` keep the 1.10 forms COMPATIBILITY.md recorded: a trailing
    side gusset in ticks, now with a trailing bottom gusset beside it."""
    from packvium.flat_bag import fits_alone, girth_admits
    inner = Dimensions.mm(300, 150, 100)
    assert girth_admits(inner, ticks(280), ticks(100), ticks(20))
    assert not girth_admits(inner, ticks(300), ticks(100), ticks(20))
    assert girth_admits(inner, ticks(300), ticks(100), ticks(20), 0, ticks(40))
    assert fits_alone(inner, Dimensions.mm(100, 100, 60), ticks(10))
    assert not fits_alone(inner, Dimensions.mm(100, 100, 60))
