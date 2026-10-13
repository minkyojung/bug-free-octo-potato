from src.data import KEY, N_PRIVATE, N_TRAIN, TARGETS, feature_columns


def test_shapes_and_columns(data):
    train, test = data
    assert len(train) == N_TRAIN and len(test) == N_PRIVATE
    assert feature_columns(train) == [c for c in test.columns if c != KEY]
    assert len(feature_columns(train)) == 284
    assert not any(t in test.columns for t in TARGETS)


def test_targets_are_binary_ints(data):
    train, _ = data
    for t in TARGETS:
        assert train[t].dtype.kind == "i" and set(train[t].unique()) == {0, 1}
