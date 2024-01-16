"""
This file contains simple models.
These are analytic and hence provide access to exact attributions.
"""
import numpy as np


def additive_syntethic_paper_regression_fn(data: np.ndarray) -> np.ndarray:
    """
    Simple analytic regression function with four features.
    Target function: x0^2 + 3*x1 + sin(x2) - x3**3 / 2
    """
    assert data.ndim > 1, 'Last dimension are feature, minimum a single sample dimension expected'
    sample_shape = data.shape[:-1]
    prediction = data[..., 0] ** 2 + 3. * data[..., 1] + np.sin((data[..., 2] - 2.)) - data[..., 3] ** 3 / 2
    return prediction.reshape(*sample_shape, 1)


def two_point_interaction_regression_fn(data: np.ndarray) -> np.ndarray:
    """
    Simple analytic regression function with four features.
    Target function: 2 * x0 - 0.5 * x1 * (x2 - 2)
    """
    assert data.ndim > 1, 'Last dimension are feature, minimum a single sample dimension expected'
    sample_shape = data.shape[:-1]
    prediction = 2. * data[..., 0] - .5 * data[..., 1] * (data[..., 2] - 2.)
    return prediction.reshape(*sample_shape, 1)


def three_point_interaction_regression_fn(data: np.ndarray) -> np.ndarray:
    """
    Simple analytic regression function with four features.
    Target function: x0 + x1 + x2 +  x0 * x1 * x2
    """
    assert data.ndim > 1, 'Last dimension are feature, minimum a single sample dimension expected'
    sample_shape = data.shape[:-1]
    prediction = data[..., 0] + data[..., 1] + data[..., 2] + data[..., 0] * data[..., 1] * data[..., 2]
    return prediction.reshape(*sample_shape, 1)


def four_point_regression_fn(x: np.ndarray) -> np.ndarray:
    """f(x) = x0 + exp(x0 + x1) + sin(x0) * x1 * x2 + x0^3 * x1 * exp(x2 + x3)"""
    assert x.ndim > 1, 'Last dimension are feature, minimum a single sample dimension expected'
    sample_shape = x.shape[:-1]
    prediction = x[..., 0] + np.exp(x[..., 0] * x[..., 1]) + np.sin(x[..., 0]) * x[..., 1] * x[..., 2] \
                 + x[..., 0]**3 * x[..., 1] * np.exp(x[..., 2] + x[..., 3])
    return prediction.reshape(*sample_shape, 1)


def one_feature_binary_classification_fn(data: np.ndarray) -> np.ndarray:
    """
    Simple classification with one features. axis=-1 is feature dimension, the rest is just passed through.
    Probability:
         class A: eps for x < 1, else 1 - eps
         class B: 1 - eps for x < 1, else eps
    """
    assert data.ndim > 1, 'Last dimension are feature, minimum a single sample dimension expected'
    eps = 0.1
    sample_shape = data.shape[:-1]
    prob_class_a = np.zeros(shape=(*sample_shape, 1))
    prob_class_a[data[..., 0] < 1] = eps
    prob_class_a[data[..., 0] >= 1] = 1 - eps
    prob_class_b = 1 - prob_class_a
    probabilities = np.stack([prob_class_a, prob_class_b], axis=-1)

    return probabilities.reshape(*sample_shape, 2)


if __name__ == '__main__':
    prob_for_zeros = one_feature_binary_classification_fn(data=np.zeros(shape=(2, 3, 4)))

    prob_for_twos = one_feature_binary_classification_fn(data=2 * np.ones(shape=(2, 3, 1)))

    samples = np.array([[1, 1, 1, 1], [1, 2, 3, 4]])
    pred_additive = additive_syntethic_paper_regression_fn(data=samples)
    print(f'original: {pred_additive}\n')
    for i in range(4):
        samples_copy = samples.copy()
        samples_copy[:, i] = 0
        imputed_pred = additive_syntethic_paper_regression_fn(data=samples_copy)
        attr = pred_additive - imputed_pred
        print(f'i = {i}\n'
              f'imputed: {imputed_pred}\n'
              f'attribution: ones: {attr[0]}, 1234: {attr[1]}')
