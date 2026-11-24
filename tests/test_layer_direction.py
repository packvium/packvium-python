"""Every module of `packvium` imports only its own layer or a lower one (R-03).

The tiers are docs/ARCHITECTURE.md's chain, `Units <- Domain <- Constraints/Algorithms/
Validation <- Packer <- Serialization/Nested <- Application adapters`, with commerce as a second
root that only the adapters may join. The source is parsed, never imported, so the test sees
lazy imports inside functions too: those are dependencies all the same. Imports under
`if TYPE_CHECKING:` are skipped, because they never execute and so cannot make a lower layer
need a higher one at run time.

A new module fails until it is placed in a tier, a new upward import fails until it is removed
or listed in `ALLOWED_UPWARD` with its reason, and a listed import that no longer exists fails
too, so the list cannot outlive what it excuses. PHP holds the same rule in LayerDirectionTest.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator

SOURCE = Path(__file__).resolve().parents[1] / "src" / "packvium"

#: Helpers every layer may use: no packing concept lives in them.
INFRASTRUCTURE = 0
UNITS = 1
DOMAIN = 2
#: Constraints, algorithms, objectives, validation and extensions share one tier, as the
#: architecture's chain puts them side by side; request admission lives here too, since
#: fixed placements and the packer both call it.
ALGORITHMS = 3
PACKER = 4
SERIALIZATION = 5
#: The Python-only intelligence surface: where the packing chain and commerce may meet.
ADAPTERS = 6
FACADE = 7
#: Commerce is its own root; see `COMMERCE_MAY_IMPORT`.
COMMERCE = -1

TIERS: dict[str, int] = {
    **dict.fromkeys(("_compat", "_canonical_json", "_request_error", "orientations", "trace",
                     "spatial_index"), INFRASTRUCTURE),
    "units": UNITS,
    **dict.fromkeys(("geometry", "compression", "flat_bag", "hull", "models", "centre_of_mass",
                     "lattice_summary", "made_to_measure", "nesting", "pallet_overhang", "loads",
                     "axle_load", "ratios", "result", "contact", "support_polygon", "explain",
                     "config", "effort",
                     # The outcome ledger's value objects: revisions and the intelligence surface
                     # both record into it, and it imports nothing of the chain.
                     "outcomes"), DOMAIN),
    **dict.fromkeys(("constraints", "bounds", "packing_sequence", "solvers", "extensions",
                     "validation", "result_validation", "fixed_placements", "policy", "rebalance",
                     "request_errors", "request_objects"), ALGORITHMS),
    "packer": PACKER,
    **dict.fromkeys(("serialization", "nested", "locks", "__main__", "artifacts", "execution",
                     "artifact_exports", "revisions", "revision_outcomes"), SERIALIZATION),
    **dict.fromkeys(("recommendations", "simulation", "holdout", "pareto"), ADAPTERS),
    "__init__": FACADE,
    "commerce": COMMERCE,
}

#: What commerce may read outside itself: two layer-free helpers, never a packing layer.
COMMERCE_MAY_IMPORT = {"_compat", "_canonical_json"}

#: Upward imports that exist today, each with why it stays. Empty is the goal.
ALLOWED_UPWARD: dict[tuple[str, str], str] = {}


def _module_name(path: Path) -> str:
    relative = path.relative_to(SOURCE).with_suffix("")
    return relative.parts[0] if relative.parts[0] == "commerce" else ".".join(relative.parts)


def _type_checking_lines(tree: ast.Module) -> set[int]:
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            for child in node.body:
                lines.update(range(child.lineno, (child.end_lineno or child.lineno) + 1))
    return lines


def _imported_modules(path: Path) -> Iterator[str]:
    """The top-level `packvium` module each import reaches, relative or absolute."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skipped = _type_checking_lines(tree)
    package = list(path.relative_to(SOURCE).parent.parts)
    for node in ast.walk(tree):
        if getattr(node, "lineno", None) in skipped:
            continue
        if isinstance(node, ast.ImportFrom) and node.level:
            base = package[:len(package) - node.level + 1]
            names = [node.module] if node.module else [alias.name for alias in node.names]
            for name in names:
                parts = base + name.split(".")
                target = parts[0]
                yield "__init__" if not base and not node.module and target not in TIERS else target
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "packvium":
            parts = node.module.split(".")[1:]
            if parts:
                yield parts[0]
            else:
                for alias in node.names:
                    yield alias.name if alias.name in TIERS else "__init__"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "packvium":
                    yield parts[1] if len(parts) > 1 else "__init__"


def _modules() -> dict[str, list[Path]]:
    modules: dict[str, list[Path]] = {}
    for path in sorted(SOURCE.rglob("*.py")):
        modules.setdefault(_module_name(path), []).append(path)
    return modules


def _upward_imports() -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for module, paths in _modules().items():
        tier = TIERS[module]
        for path in paths:
            for imported in _imported_modules(path):
                if imported == module:
                    continue
                target = TIERS[imported]
                if tier == COMMERCE:
                    upward = imported not in COMMERCE_MAY_IMPORT
                elif target == COMMERCE:
                    upward = tier < ADAPTERS
                else:
                    upward = target > tier
                if upward:
                    found.add((module, imported))
    return found


def test_every_module_is_placed_in_a_tier():
    unplaced = sorted(set(_modules()) - set(TIERS))
    assert not unplaced, f"place these modules in TIERS: {unplaced}"


def test_no_tier_names_a_module_that_does_not_exist():
    assert not sorted(set(TIERS) - set(_modules()))


def test_no_module_imports_a_higher_layer():
    unexplained = sorted(_upward_imports() - set(ALLOWED_UPWARD))
    assert not unexplained, f"upward imports (importer, imported): {unexplained}"


def test_every_allowed_upward_import_still_exists():
    stale = sorted(set(ALLOWED_UPWARD) - _upward_imports())
    assert not stale, f"remove these from ALLOWED_UPWARD: {stale}"


def test_absolute_import_checks_every_name_and_the_package_facade(tmp_path, monkeypatch):
    monkeypatch.setitem(globals(), "SOURCE", tmp_path)
    module = tmp_path / "units.py"
    module.write_text(
        "from packvium import models, solvers\nimport packvium\nfrom packvium import validate\n",
        encoding="utf-8",
    )
    assert list(_imported_modules(module)) == ["models", "solvers", "__init__", "__init__"]
