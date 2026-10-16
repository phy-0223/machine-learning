import numpy as np

from ml_course.gradient_descent import (
    finite_difference_gradient,
    make_regression_data,
    mse_and_gradients,
    train_linear_regression,
)


def test_analytical_gradient_matches_finite_difference() -> None:
    X, y = make_regression_data(samples=20, noise=0.0)
    weights = np.array([0.3, -0.7])
    bias = 0.2
    _, analytical_weights, analytical_bias = mse_and_gradients(X, y, weights, bias)
    numerical_weights, numerical_bias = finite_difference_gradient(X, y, weights, bias)
    assert np.allclose(analytical_weights, numerical_weights, atol=1e-6)
    assert np.isclose(analytical_bias, numerical_bias, atol=1e-6)


def test_gradient_descent_recovers_known_parameters() -> None:
    X, y = make_regression_data(samples=500, noise=0.05)
    model = train_linear_regression(X, y)
    assert model.loss_history[-1] < model.loss_history[0] * 0.01
    assert np.allclose(model.weights, [3.0, -2.0], atol=0.03)
    assert np.isclose(model.bias, 0.5, atol=0.03)

