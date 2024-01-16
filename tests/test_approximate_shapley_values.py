import numpy as np

from src.hoocs.approximate_shapley_values import ApproximateShapleyValues, shapley_normalization
from src.hoocs.imputers import simple_imputers
from tests import simple_models
from tests.test_kernelshap import check_efficiency

from numpy.testing import assert_allclose

import pytest

def test_shapley_normalization():
    assert shapley_normalization(1, 0) == 1
    assert shapley_normalization(2, 1) == 0.5
    assert shapley_normalization(7, 0) == 1 / 7
    assert shapley_normalization(4, 1) == 1 / 12
    assert shapley_normalization(4, 3) == 1 / 4
    assert shapley_normalization(5, 2) == 1 / 30


class TestApproximateShapleyValues:
    @pytest.mark.skip(reason="no way of currently testing this")
    def test_preddiff_two_point(self):
        """Use simple four feature regression model to test one and two point attributions."""
        imputer = simple_imputers.ConstantValueImputer(constant=0)
        explainer = ApproximateShapleyValues(model_fn=simple_models.two_point_interaction_regression_fn,
                                             imputer=imputer, data=np.ones((3, 2, 4)), n_eval=3,
                                             cardinality_coalitions=[-1])

        data = np.array([[1, 1, 1, 1], [1, 2, 3, 4]])
        seg = [4, 1, 2, 10]
        segmentation = np.stack([seg for _ in range(2)])
        target_features = set(segmentation.flatten())

        dict_attributions = explainer.attribution(data=data, segmentation=segmentation,
                                                  target_features=target_features)

        # check_efficiency(dict_attributions=dict_attributions, target_features=target_features)

        assert_allclose(dict_attributions['4'].reshape(2), np.array([2, 2]))
        assert_allclose(dict_attributions['1'].reshape(2), np.array([0.5, -1]))
        assert_allclose(dict_attributions['2'].reshape(2), np.array([-0.5, -3]))
        assert_allclose(dict_attributions['10'].reshape(2), np.array([0, 0]))

    @pytest.mark.skip(reason="no way of currently testing this")
    def test_shapley_two_point(self):
        """Use simple four feature regression model to test one and two point attributions."""
        imputer = simple_imputers.ConstantValueImputer(constant=0)
        # explainer_none = ApproximateShapleyValues(
        #     model_fn=simple_models.two_point_interaction_regression_fn, imputer=imputer,
        #     data=np.ones((3, 2, 4)), n_eval=5_000, cardinality_coalitions=None
        # )

        explainer_explicit = ApproximateShapleyValues(
            model_fn=simple_models.two_point_interaction_regression_fn, imputer=imputer,
            data=np.ones((3, 2, 4)), n_eval=10000, cardinality_coalitions=[-1, -2, -3]
        )

        for explainer in [explainer_explicit]:
            data = np.array([[1, 1, 1, 1], [1, 2, 3, 4]])
            seg = [4, 1, 2, 10]
            segmentation = np.stack([seg for _ in range(2)])

            target_features = set(segmentation.flatten())
            dict_attributions = explainer.attribution(data=data, segmentation=segmentation,
                                                      target_features=target_features)
            check_efficiency(dict_attributions=dict_attributions, target_features=target_features)

            assert_allclose(dict_attributions['1'].reshape(2), [0.75, 0.5], atol=0.1)
            assert_allclose(dict_attributions['2'].reshape(2), [-0.25, -1.5], atol=0.1)
