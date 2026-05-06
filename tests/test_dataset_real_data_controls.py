import pytest

from src.data import dataset as dataset_module


def test_create_lm_dataset_does_not_silently_fall_back_to_dummy(monkeypatch):
    monkeypatch.setattr(dataset_module, "HAS_DATASETS", False)

    with pytest.raises(ImportError):
        dataset_module.create_lm_dataset()


def test_create_lm_dataset_allows_explicit_dummy_fallback(monkeypatch):
    monkeypatch.setattr(dataset_module, "HAS_DATASETS", False)

    dataset = dataset_module.create_lm_dataset(allow_dummy_fallback=True)

    batch = next(iter(dataset.batch(2)))
    assert batch["input_ids"].shape == (2, 512)

