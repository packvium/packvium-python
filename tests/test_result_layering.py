import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src" / "packvium"

# The result object is Domain layer: it may read domain values and pure arithmetic, never the
# constraint, solver or validation layers above it (R-03).
RESULT_MAY_IMPORT = {"_compat", "axle_load", "loads", "models", "ratios", "units"}


def _packvium_imports(module: str) -> set[str]:
    tree = ast.parse((SOURCE / module).read_text())
    return {
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module
    }


def test_the_result_object_imports_nothing_above_the_domain_layer():
    assert _packvium_imports("result.py") <= RESULT_MAY_IMPORT


def test_the_modules_the_result_object_reads_import_no_higher_layer():
    for module in ("axle_load", "loads", "ratios"):
        assert _packvium_imports(f"{module}.py") <= {"models", "geometry", "loads", "_compat"}, module
