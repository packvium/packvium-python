"""Lookup optimizations preserve public model values and compatibility behavior."""
import pytest

from packvium.commerce.catalog import (
    CatalogRegistry, CatalogSnapshot, CatalogVersionNotFoundError,
    CatalogEntryNotFoundError, ItemMaster, CartonMaster,
)
from packvium.commerce.rating import CarrierRegistry, TariffNotFoundError


def make_item(id="widget"):
    return ItemMaster(id, (80, 60, 40), 250)


def make_snapshot():
    return CatalogSnapshot(items=(make_item(),),
                           cartons=(CartonMaster("box-s", (180, 140, 100), 5000),))


def test_pinned_history_lookup_preserves_boundaries_rollbacks_and_numeric_compatibility():
    registry = CatalogRegistry("lookup")
    snapshot = make_snapshot()
    for number in range(1, 65):
        registry.publish(snapshot, effective_at=number % 7, published_at=number)
    for number in (1, 2, 32, 64):
        assert registry.resolve(version=number, resolved_at=100).reference.version == number
    for number in (-10, 0, 65, 10 ** 100, 1.5, "1"):
        with pytest.raises(CatalogVersionNotFoundError):
            registry.resolve(version=number, resolved_at=100)
    assert registry.resolve(version=True, resolved_at=100).reference.version == 1
    assert registry.resolve(version=2.0, resolved_at=100).reference.version == 2
    rollback = registry.rollback(1, published_at=100, effective_at=6)
    assert rollback.number == 65
    assert registry.resolve(version=65, resolved_at=100).snapshot is snapshot
    assert registry.resolve(as_of=6, resolved_at=100).reference.version == 65
    assert registry.resolve(as_of=0, resolved_at=100).reference.version == 63


def test_snapshot_lookup_indexes_do_not_change_value_copy_or_serialization():
    import copy
    from dataclasses import asdict, fields
    import pickle

    snapshot = make_snapshot()
    before = asdict(snapshot)
    before_hash = hash(snapshot)
    assert snapshot.item('widget') is snapshot.items[0]
    assert snapshot.carton('box-s') is snapshot.cartons[0]
    assert asdict(snapshot) == before
    assert hash(snapshot) == before_hash
    assert [field.name for field in fields(snapshot)] == ['items', 'cartons', 'pallets', 'exclusions', 'overrides']
    for copied in (copy.copy(snapshot), copy.deepcopy(snapshot), pickle.loads(pickle.dumps(snapshot))):
        assert copied == snapshot
        assert copied.item('widget') == snapshot.items[0]
        assert asdict(copied) == before
    restored = CatalogSnapshot.__new__(CatalogSnapshot)
    restored.__setstate__({field.name: getattr(snapshot, field.name) for field in fields(snapshot)})
    assert restored.item('widget') == snapshot.items[0]
    # A snapshot pickled before the fields were normalised to tuples holds lists; restoring it
    # gives the tuples a new snapshot has, so the restored history cannot be edited either.
    older = CatalogSnapshot.__new__(CatalogSnapshot)
    older.__setstate__({field.name: list(getattr(snapshot, field.name)) for field in fields(snapshot)})
    assert older.item('widget') == snapshot.items[0]
    assert older.carton('box-s') == snapshot.cartons[0]
    assert all(type(getattr(older, field.name)) is tuple for field in fields(older))


def test_snapshot_lookup_preserves_custom_input_behavior_and_detaches_the_callers_list():
    entries = [make_item('first')]
    snapshot = CatalogSnapshot(items=entries)
    assert snapshot.item('first') is entries[0]
    entries.append(make_item('second'))
    assert snapshot.items == (entries[0],)
    with pytest.raises(CatalogEntryNotFoundError):
        snapshot.item('second')
    class MutableEntry:
        id = 'old'
    entry = MutableEntry()
    custom = CatalogSnapshot(items=(entry,))
    assert custom.item('old') is entry
    entry.id = 'new'
    assert custom.item('new') is entry
    with pytest.raises(CatalogEntryNotFoundError):
        custom.item('old')


def test_indexed_snapshot_lookup_keeps_rejection_for_non_string_identifiers():
    snapshot = make_snapshot()
    snapshot.item('widget')
    with pytest.raises(CatalogEntryNotFoundError):
        snapshot.item(1)


def test_pinned_and_effective_history_keep_numeric_and_date_boundaries():
    registry = CarrierRegistry()
    for index in range(1, 65):
        registry.publish('acme-freight', 'ground', effective_at=index % 7,
                         dimensional_weight_divisor=5000,
                         cost_per_dimensional_kg_minor={'zone-1': 500})
    for version in (1, 32, 64, 2.0, True):
        assert registry.tariff('acme-freight', 'ground', version).version == version
    for version in (-1, 0, 65, 10 ** 100, 1.5, '1'):
        with pytest.raises(TariffNotFoundError):
            registry.tariff('acme-freight', 'ground', version)
    assert registry.effective_tariff('acme-freight', 'ground', as_of=6).version == 62
    assert registry.effective_tariff('acme-freight', 'ground', as_of=0).version == 63
    with pytest.raises(TariffNotFoundError):
        registry.effective_tariff('acme-freight', 'ground', as_of=-1)


def test_published_history_cannot_be_edited_through_the_callers_or_the_returned_objects():
    import copy
    import pickle
    from packvium.commerce.rating import AccessorialCharge
    dimensions = [80, 60, 40]
    master = ItemMaster("widget", dimensions, 250)
    dimensions[0] = 1
    assert master.dimensions_mm == (80, 60, 40)

    rates = {"zone-1": 500}
    surcharges = {"liftgate": AccessorialCharge("liftgate", flat_charge_minor=900)}
    registry = CarrierRegistry()
    registry.publish("acme-freight", "ground", effective_at=0, dimensional_weight_divisor=5000,
                     cost_per_dimensional_kg_minor=rates, accessorials=surcharges)
    rates["zone-1"] = 1
    pinned = registry.tariff("acme-freight", "ground", 1)
    assert pinned.cost_per_dimensional_kg_minor == {"zone-1": 500}
    with pytest.raises(TypeError):
        pinned.cost_per_dimensional_kg_minor["zone-1"] = 1
    with pytest.raises(TypeError):
        pinned.accessorials["extra"] = surcharges["liftgate"]
    for restored in (copy.deepcopy(pinned), pickle.loads(pickle.dumps(pinned))):
        assert restored == pinned
        with pytest.raises(TypeError):
            restored.cost_per_dimensional_kg_minor["zone-1"] = 1
