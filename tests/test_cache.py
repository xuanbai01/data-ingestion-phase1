"""
Tests for the feature cache layer in database/db.py.

Covers the four behaviors the rest of the pipeline depends on:
- round-trip insert / load preserves all field values, including embeddings
- schema_version mismatches are misses
- embedder_model mismatches are misses (the silent-poisoning guard)
- INSERT OR REPLACE actually replaces an existing row
"""
from database import db


SAMPLE_REVIEW = {
    "reviewer_name": "alice",
    "date": "2026-01-01",
    "body": "the checkout keeps crashing",
}

SAMPLE_FEATURES = {
    "polarity": -0.42,
    "subjectivity": 0.31,
    "aspects": ["checkout", "crash"],
    "entities": [{"text": "Stripe", "label": "ORG"}],
    "emotion": "anger",
    "urgency": 0.85,
    "embedding": [0.1, -0.2, 0.3, 0.4, -0.5],
}


def test_save_and_load_round_trip(temp_db):
    key = db.compute_cache_key(SAMPLE_REVIEW)
    db.save_features_batch([(key, SAMPLE_FEATURES)], schema_version=1, embedder_model="m1")

    loaded = db.load_features_batch([key], schema_version=1, embedder_model="m1")

    assert key in loaded
    f = loaded[key]
    assert f["polarity"] == SAMPLE_FEATURES["polarity"]
    assert f["subjectivity"] == SAMPLE_FEATURES["subjectivity"]
    assert f["aspects"] == SAMPLE_FEATURES["aspects"]
    assert f["entities"] == SAMPLE_FEATURES["entities"]
    assert f["emotion"] == SAMPLE_FEATURES["emotion"]
    assert f["urgency"] == SAMPLE_FEATURES["urgency"]
    # Embedding round-trips through float32; allow small precision loss.
    assert len(f["embedding"]) == len(SAMPLE_FEATURES["embedding"])
    for got, want in zip(f["embedding"], SAMPLE_FEATURES["embedding"]):
        assert abs(got - want) < 1e-6


def test_schema_version_mismatch_is_miss(temp_db):
    key = db.compute_cache_key(SAMPLE_REVIEW)
    db.save_features_batch([(key, SAMPLE_FEATURES)], schema_version=1, embedder_model="m1")

    loaded = db.load_features_batch([key], schema_version=2, embedder_model="m1")
    assert loaded == {}


def test_embedder_mismatch_is_miss(temp_db):
    """Even at the same schema version, a different embedder model is treated as a miss."""
    key = db.compute_cache_key(SAMPLE_REVIEW)
    db.save_features_batch([(key, SAMPLE_FEATURES)], schema_version=1, embedder_model="m1")

    loaded = db.load_features_batch([key], schema_version=1, embedder_model="m2")
    assert loaded == {}


def test_overwrite_replaces_existing_row(temp_db):
    key = db.compute_cache_key(SAMPLE_REVIEW)
    first = {**SAMPLE_FEATURES, "polarity": 0.1, "embedding": [0.0]}
    second = {**SAMPLE_FEATURES, "polarity": 0.9, "embedding": [1.0]}

    db.save_features_batch([(key, first)], schema_version=1, embedder_model="m1")
    db.save_features_batch([(key, second)], schema_version=1, embedder_model="m1")

    loaded = db.load_features_batch([key], schema_version=1, embedder_model="m1")
    assert loaded[key]["polarity"] == 0.9
    assert abs(loaded[key]["embedding"][0] - 1.0) < 1e-6


def test_missing_keys_return_empty(temp_db):
    loaded = db.load_features_batch(["nonexistent-hash"], schema_version=1, embedder_model="m1")
    assert loaded == {}


def test_empty_input_is_safe(temp_db):
    assert db.load_features_batch([], schema_version=1, embedder_model="m1") == {}
    db.save_features_batch([], schema_version=1, embedder_model="m1")  # should not raise


def test_clear_cache_drops_all(temp_db):
    key = db.compute_cache_key(SAMPLE_REVIEW)
    db.save_features_batch([(key, SAMPLE_FEATURES)], schema_version=1, embedder_model="m1")

    deleted = db.clear_feature_cache()
    assert deleted == 1
    assert db.load_features_batch([key], schema_version=1, embedder_model="m1") == {}


def test_clear_cache_keeps_current_version(temp_db):
    """clear_feature_cache(version) drops everything *except* that version."""
    key1 = db.compute_cache_key(SAMPLE_REVIEW)
    key2 = db.compute_cache_key({**SAMPLE_REVIEW, "body": "different review"})

    db.save_features_batch([(key1, SAMPLE_FEATURES)], schema_version=1, embedder_model="m1")
    db.save_features_batch([(key2, SAMPLE_FEATURES)], schema_version=2, embedder_model="m1")

    deleted = db.clear_feature_cache(schema_version=2)
    assert deleted == 1
    # v2 row stays, v1 row is gone
    assert db.load_features_batch([key1], schema_version=1, embedder_model="m1") == {}
    assert key2 in db.load_features_batch([key2], schema_version=2, embedder_model="m1")


def test_cache_key_is_stable_across_dict_order(temp_db):
    """compute_cache_key depends on values, not dict insertion order."""
    a = {"reviewer_name": "x", "date": "2026-01-01", "body": "hello"}
    b = {"body": "hello", "date": "2026-01-01", "reviewer_name": "x"}
    assert db.compute_cache_key(a) == db.compute_cache_key(b)


def test_cache_key_handles_none_body(temp_db):
    """A None body shouldn't crash hashing."""
    review = {"reviewer_name": "x", "date": "2026-01-01", "body": None}
    key = db.compute_cache_key(review)
    assert isinstance(key, str)
    assert len(key) == 32  # MD5 hex length


def test_chunked_lookup_handles_large_keysets(temp_db):
    """Stay correct when the input list exceeds the SQLite parameter chunk size."""
    items = []
    keys = []
    for i in range(1200):  # > _PARAM_CHUNK (500)
        review = {"reviewer_name": f"u{i}", "date": "2026-01-01", "body": f"body {i}"}
        key = db.compute_cache_key(review)
        keys.append(key)
        items.append((key, {**SAMPLE_FEATURES, "polarity": i / 1000.0}))

    db.save_features_batch(items, schema_version=1, embedder_model="m1")
    loaded = db.load_features_batch(keys, schema_version=1, embedder_model="m1")

    assert len(loaded) == 1200
    # Sample one to confirm the value rode through correctly.
    assert abs(loaded[keys[500]]["polarity"] - 0.5) < 1e-6
