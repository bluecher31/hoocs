import numpy as np
from numpy.testing import assert_allclose

from src.hoocs import shapley
from src.hoocs.imputers import simple_imputers
from tests import simple_models

import pytest


def test_generate_footprints_shapley():
    n_coalitions = 5
    list_footprint = shapley.generate_footprints_shapley(['empty_set', '1', '2', '1^2'], n_coalitions=n_coalitions)
    for i in range(n_coalitions):
        f0 = list_footprint[0][:, i]
        assert np.alltrue(f0 == np.array([+1., -1., 0, 0])) or np.alltrue(f0 == np.array([0, 0, +1, -1])), \
            'Incorrect footprint main effect 1'

        f1 = list_footprint[1][:, i]
        assert np.alltrue(f1 == np.array([+1., 0, -1., 0])) or np.alltrue(f1 == np.array([0, +1, 0, -1])), \
            'Incorrect footprint main effect 2'

        f2 = list_footprint[2][:, i]
        assert np.alltrue(f2 == np.array([-1., 1, 1, -1.])), 'Incorrect footprint 2-point interaction 1^2'


class TestShapley:
    def test_attribution_additive_regression(self):
        """Test additive attributions using the maximum_interaction capability."""
        imputer = simple_imputers.ConstantValueImputer(constant=0)
        explainer = shapley.ShapleyValues(model_fn=simple_models.additive_syntethic_paper_regression_fn,
                                          imputer=imputer, data=np.ones((3, 2, 4)), n_eval=15)

        data = np.array([[1, 1, 1, 1], [1, 2, 3, 4]])
        seg = [7, 1, 2, 8]
        segmentation = np.stack([seg for _ in range(2)])

        dict_target_atttributions = {'1': [3, 6], '2': [0.06782644, 1.75076841], '7': [1, 1], '8': [-0.5, -32]}

        for target in np.unique(segmentation):
            dict_attribution = explainer.attribution(data=data, segmentation=segmentation,
                                                     target_features={int(target)})
            assert_allclose(dict_attribution[str(target)].flatten(), dict_target_atttributions[str(target)])

    def test_attribution_interactions_regression(self):
        """Use simple four feature regression model to test one and two point attributions."""
        imputer = simple_imputers.ConstantValueImputer(constant=0)
        explainer = shapley.ShapleyValues(
            model_fn=simple_models.two_point_interaction_regression_fn, imputer=imputer,
            data=np.ones((3, 2, 4)), n_eval=10_000
        )

        data = np.array([[1, 1, 1, 1], [1, 2, 3, 4]])
        seg = [4, 1, 2, 10]
        segmentation = np.stack([seg for _ in range(2)])

        dict_attributions = explainer.attribution(data=data, segmentation=segmentation,
                                                  target_features={1, 2})
        assert_allclose(dict_attributions['1'].reshape(2), [0.75, 0.5], atol=0.05)
        assert_allclose(dict_attributions['2'].reshape(2), [-0.25, -1.5], atol=0.05)
        # sign convention and prefactors for expected results according to
        # https://christophm.github.io/interpretable-ml-book/shap.html#shap-interaction-values
        # TODO: find factor 1/2 and change sign
        assert_allclose(-0.5*dict_attributions['1^2'].reshape(2), [-0.25, -1.5])

        # TODO: add a test for higher-order attributions, prefactor is potentially incorrect
        #  use this faith-shap resource https://arxiv.org/pdf/2203.00870.pdf
        # explainer.interaction_depth = 2
        # dict_attributions = explainer.attribution(data=data, segmentation=segmentation,
        #                                           target_features=np.array([0, 1, 2, 10]))
        # assert np.allclose(dict_attributions['0'].reshape(2), np.array([2, 2]))
        # assert np.allclose(dict_attributions['1'].reshape(2), np.array([0.75, 0.5]))
        # assert np.allclose(dict_attributions['2'].reshape(2), np.array([-0.25, -1.5]))
        # assert np.allclose(dict_attributions['10'].reshape(2), np.array([0, 0]))
        #
        # # sign convention from
        # # https://christophm.github.io/interpretable-ml-book/shap.html#shap-interaction-values
        # assert np.allclose(dict_attributions['0^10'].reshape(2), np.array([0, 0]))
        # assert np.allclose(dict_attributions['1^2'].reshape(2), np.array([-0.25, -1.5]))


class TestShapleySubsamplingCardinalities():
    """This class test results around subsampling cardinalities."""

    def test_preddiff_additive(self):
        """
        Using only coalitions with |S| = n - 1, this correspond to PredDiff up to rescaling.
        """
        imputer = simple_imputers.ConstantValueImputer(constant=0)
        f = simple_models.four_point_regression_fn
        a = np.array([1, 2, -1, 0.5])  # f(a) = 7.91917545
        b = np.array([2, -1, 1.5, -1])  # f(b) = -12.41838102
        data = np.stack([a, b])
        # f(a\1) = 1,               f(b\1) = 1
        # m^a_1 = 6.91917545        m^b_1 = -13.41838102
        # f(a\2) = 2,               f(b\2) = 3
        # m^a_2 = 5.91917545        m^b_2 = -15.41838102
        # f(a\3) = 11.68649864,     f(b\3) = -0.80770025
        # m^a_3 = -3.76732319       m^b_3 = -11.61068077
        # f(a\4) = 7.44187301,      f(b\4) = -35.08212342
        # m^a_4 = 0.477302439       m^b_4 = 22.663742400000004
        dict_target_attributions = {'1': [6.91917545, -13.41838102], '2': [5.91917545, -15.41838102],
                                    '3': [-3.76732319, -11.61068077], '4': [0.477302439, 22.663742400000004]}
        # imp_data = data.copy()
        # imp_data[:, 3] = 0
        # print(imp_data)
        # print('result: \n', f(imp_data))
        explainer = shapley.ShapleyValues(model_fn=f, imputer=imputer, data=np.ones((3, 2, 4)), n_eval=15,
                                          cardinality_coalitions=[-1])
        # explainer.interaction_depth = 2

        segmentation = np.stack([np.arange(1, 5) for _ in range(2)])
        for key in dict_target_attributions:
            dict_attributions = explainer.attribution(data=data, segmentation=segmentation,
                                                      target_features={int(key)})

            assert_allclose(dict_attributions[key][:, 0], dict_target_attributions[key], atol=0.5,
                            err_msg=f'Incorrect attributions for key = {key}\n'
                                    f'x = attribution, y = target')

    @pytest.mark.skip(reason="no way of currently testing this")
    def test_preddiff_interactions(self):
        mu_data = np.array([-2, 5, 7, 10])
        cov_data = np.eye(4)
        imputer = simple_imputers.GaussianNoiseImputer(mu_data, cov_data)

        explainer = shapley.ShapleyValues(model_fn=simple_models.three_point_interaction_regression_fn,
                                          imputer=imputer, data=np.ones((3, 2, 4)), n_eval=5_000,
                                          cardinality_coalitions=[-1])

        data = np.array([[1, 4, 3, -10], [2, -1, 6, 4]])
        seg = [4, 1, 2, 10]
        segmentation = np.stack([seg for _ in range(2)])

        # explainer.interaction_depth = 3
        target_features = {4, 1, 2}
        dict_attributions = explainer.attribution(data=data, segmentation=segmentation,
                                                  target_features=target_features)
        dict_target_attributions = {'4': [39, -20], '1': [-4, -78], '2': [-20, 1],
                                    '1^2': [-4, -12], '1^2^4': [12, 24]}
        for key in dict_target_attributions:
            assert_allclose(dict_attributions[key][:, 0], dict_target_attributions[key], atol=0.5,
                            err_msg=f'Incorrect attributions for key = {key}\n'
                                    f'x = attribution, y = target')

    def test_shapley_original(self):
        """Use simple four feature regression model to test one and two point attributions."""
        imputer = simple_imputers.ConstantValueImputer(constant=0)
        data = np.array([[1, 1, 1, 1], [1, 2, 3, 4]])
        seg = [4, 1, 2, 10]
        segmentation = np.stack([seg for _ in range(2)])

        for cardinalities in [[1, 2, -1, -2, -3, -4], [1, 2, 3, 4], [-1, -2, -3, -4], [1, 2, -1, -2]]:
            explainer = shapley.ShapleyValues(model_fn=simple_models.two_point_interaction_regression_fn, imputer=imputer,
                                              data=np.ones((3, 2, 4)), n_eval=2_000, cardinality_coalitions=cardinalities)

            dict_attributions = explainer.attribution(data=data, segmentation=segmentation,
                                                      target_features={1, 2})
            assert_allclose(dict_attributions['1'].reshape(2), [0.75, 0.5], atol=0.09)
            assert_allclose(dict_attributions['2'].reshape(2), [-0.25, -1.5], atol=0.09)
            # sign convention and prefactors for expected results according to
            # https://christophm.github.io/interpretable-ml-book/shap.html#shap-interaction-values
            # TODO: find factor 1/2 and change sign
            assert_allclose(-0.5 * dict_attributions['1^2'].reshape(2), [-0.25, -1.5])