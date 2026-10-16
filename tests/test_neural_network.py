import numpy as np

from ml_course.neural_network import train_xor, xor_data


def test_two_layer_network_learns_xor() -> None:
    X, y = xor_data()
    network = train_xor()
    assert np.array_equal(network.predict(X), y.astype(int))
    assert network.loss_history[-1] < 0.05

