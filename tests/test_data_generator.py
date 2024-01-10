import numpy as np

from deep_uncertainty.data_generator import DataGenerator


def test_random_data_generator_train_test_val_split_is_valid():
    def data_gen_function():
        return np.random.random(100), np.random.random(100)

    data_splits = DataGenerator.generate_train_test_val_split(
        data_gen_function, {}, split_pcts=[0.8, 0.1, 0.1]
    )
    assert set(data_splits.keys()) == {"X_train", "X_val", "X_test", "y_train", "y_val", "y_test"}
    num_train = len(data_splits["X_train"])
    num_val = len(data_splits["X_val"])
    num_test = len(data_splits["X_test"])

    assert num_train == 80
    assert num_val == 10
    assert num_test == 10
