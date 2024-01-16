"""
This file provides test PredDiff attribution up to third order.

To this end we provide a regression toy function as model and fixed test points.
imputer_fn is deterministic.


We test different functions.
First we test each main effect individually.
Next, two-point joined and shielded effect
Lastly, all three point effect (unshielded)
"""
import numpy as np

from src.hoocs import helper_methods
from src.hoocs import preddiff
from src.hoocs.imputers import simple_imputers

from tests import simple_models, helper

from numpy.testing import assert_allclose


class TestPredDiff:
    def test_attribution_additive_regression(self):
        """Test additive attributions using the maximum_interaction capability."""
        helper.check_additivity(explainer_cls=preddiff.PredDiff)

    def test_attribution_two_point_interaction(self):
        """Use simple four feature regression model to test one and two point attributions."""
        imputer = simple_imputers.ConstantValueImputer(constant=0)
        explainer = preddiff.PredDiff(model_fn=simple_models.two_point_interaction_regression_fn,
                                      imputer=imputer, data=np.ones((3, 2, 4)), n_eval=3)

        data = np.array([[1, 1, 1, 1], [1, 2, 3, 4]])
        seg = [4, 1, 2, 10]
        segmentation = np.stack([seg for _ in range(2)])
        target_features = set(segmentation.flatten())

        explainer.interaction_depth = 2
        dict_attributions = explainer.attribution(data=data, segmentation=segmentation,
                                                  target_features=target_features)
        assert np.allclose(dict_attributions['4'].reshape(2), np.array([2, 2]))
        assert np.allclose(dict_attributions['1'].reshape(2), np.array([0.5, -1]))
        assert np.allclose(dict_attributions['2'].reshape(2), np.array([-0.5, -3]))
        assert np.allclose(dict_attributions['10'].reshape(2), np.array([0, 0]))

        assert np.allclose(dict_attributions['4^10'].reshape(2), np.array([0, 0]))
        assert np.allclose(dict_attributions['1^2'].reshape(2), np.array([0.5, 3]))

    def test_three_point_interaction(self):
        mu_data = np.array([-2, 5, 7, 10])
        cov_data = np.eye(4)
        imputer = simple_imputers.GaussianNoiseImputer(mu_data, cov_data)

        explainer = preddiff.PredDiff(model_fn=simple_models.three_point_interaction_regression_fn,
                                      imputer=imputer, data=np.ones((3, 2, 4)), n_eval=50_000)

        data = np.array([[1, 4, 3, -10], [2, -1, 6, 4]])
        seg = [4, 1, 2, 10]
        segmentation = np.stack([seg for _ in range(2)])

        explainer.interaction_depth = 3
        target_features = {4, 1, 2, 10}
        dict_attributions = explainer.attribution(data=data, segmentation=segmentation,
                                                  target_features=target_features)
        dict_target_attributions = {'4': [39, -20], '1': [-4, -78], '2': [-20, 1], '10': [0, 0],
                                    '1^2': [-4, -12], '1^2^4': [12, 24]}
        for key in dict_target_attributions:
            assert_allclose(dict_attributions[key][:, 0], dict_target_attributions[key], atol=0.5,
                            err_msg=f'Incorrect attributions for key = {key}\n'
                                    f'x = attribution, y = target')

    def test_attribution_classification(self):
        """Use simple one feature classification"""
        imputer = simple_imputers.IdentityImputer()
        explainer = preddiff.PredDiff(model_fn=simple_models.one_feature_binary_classification_fn,
                                      imputer=imputer, data=np.ones((3, 2, 4)), n_eval=3)

        data = np.array([[0, 1, 0, 0], [2, 2, 2, 2], [0, 1, 0, 0]])
        seg = [4, 1, 2, 10]
        segmentation = np.stack([seg for _ in range(3)])

        # check with identity imputer
        dict_attribution_0 = explainer.attribution(data=data, segmentation=segmentation,
                                                   target_features={4})
        assert np.allclose(dict_attribution_0['4'], np.zeros((3, 2)))

        # zero imputer
        imputer = simple_imputers.ConstantValueImputer(constant=0)
        explainer = preddiff.PredDiff(model_fn=simple_models.one_feature_binary_classification_fn,
                                      imputer=imputer, data=np.ones((3, 2, 4)), n_eval=3)

        dict_attribution_0 = explainer.attribution(data=data, segmentation=segmentation,
                                                   target_features={4})
        eps = 0.1
        assert np.allclose(dict_attribution_0['4'],
                           np.array([[0, 0],
                                     [np.log2((1 - eps) / eps), np.log2(eps / (1 - eps))],
                                     [0, 0]]), atol=5e-3)

    def test_1point_approximation(self):
        """
        Sanity check for four point function used to validate KernelSHAP
        Using only coalitions with |S| = 1, this correspond to PredDiff up to rescaling.
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
        explainer = preddiff.PredDiff(model_fn=f, imputer=imputer, data=np.ones((3, 2, 4)), n_eval=15)
        explainer.interaction_depth = 2

        segmentation = np.stack([np.arange(1, 5) for _ in range(2)])
        dict_attributions = explainer.attribution(data=data, segmentation=segmentation,
                                                  target_features=set(segmentation.flatten()))
        for key in dict_target_attributions:
            assert_allclose(dict_attributions[key][:, 0], dict_target_attributions[key], atol=0.5,
                            err_msg=f'Incorrect attributions for key = {key}\n'
                                    f'x = attribution, y = target')


def test_generate_feature_interaction_algebra():
    # one-point interaction
    list_occluded_features = helper_methods.generate_feature_interaction_algebra(target_features={1})
    assert list_occluded_features == ['empty_set', '1'], 'Incorrect 1-point feature_interaction'

    # two-point interactions
    list_occluded_features = helper_methods.generate_feature_interaction_algebra(target_features={1, 2})
    assert list_occluded_features == ['empty_set', '1', '2', '1^2'], 'Incorrect 2-point feature_interaction'

    # three-point interactions
    list_occluded_features = helper_methods.generate_feature_interaction_algebra(target_features={1, 2, 3})
    assert list_occluded_features == ['empty_set', '1', '2', '3', '1^2', '1^3', '2^3', '1^2^3'], \
        'Incorrect 3-point feature_interaction'

    # three-point interactions
    list_occluded_features = helper_methods.generate_feature_interaction_algebra(
        target_features={1, 2, 3, 4}, interaction_depth=2
    )

    list_expected_unsorted = ['empty_set', '1', '2', '3', '4', '1^2', '1^3', '1^4', '2^3', '2^4', '3^4']
    assert len(list_occluded_features) == len(list_expected_unsorted)
    assert set(list_occluded_features) == set(list_expected_unsorted), \
        'Incorrect 4-point feature_interaction with a maximum of two features interacting.'


def test_generate_footprints():
    # one-point interaction
    list_footprint = helper_methods.generate_footprints(list_occluded_features=['empty_set', '1'])
    assert np.alltrue(list_footprint[0] == np.array([+1., -1.])), 'Incorrect footprint main effect 1'

    # two-point interactions
    list_footprint = helper_methods.generate_footprints(list_occluded_features=['empty_set', '1', '2', '1^2'])
    assert np.alltrue(list_footprint[0] == np.array([+1., -1., 0, 0])), 'Incorrect footprint main effect 1'
    assert np.alltrue(list_footprint[1] == np.array([+1., 0, -1., 0])), 'Incorrect footprint main effect 2'
    assert np.alltrue(list_footprint[2] == np.array([-1., 1, 1, -1.])), 'Incorrect footprint 2-point interaction 1^2'

    # three-point interactions
    list_footprint = helper_methods.generate_footprints(
        list_occluded_features=['empty_set', '1', '2', '3', '1^2', '1^3', '2^3', '1^2^3'])
    assert np.alltrue(list_footprint[0] == np.array([+1., -1., 0, 0, 0, 0, 0, 0])), 'Incorrect footprint main effect 1'
    assert np.alltrue(list_footprint[1] == np.array([+1., 0., -1, 0, 0, 0, 0, 0])), 'Incorrect footprint main effect 2'
    assert np.alltrue(list_footprint[2] == np.array([+1., 0, 0, -1, 0, 0, 0, 0])), 'Incorrect footprint main effect 3'

    assert np.alltrue(list_footprint[3] == np.array([-1., +1., +1, 0, -1, 0, 0, 0])), 'Incorrect footprint 1^2'
    assert np.alltrue(list_footprint[4] == np.array([-1., +1., 0, +1, 0, -1, 0, 0])), 'Incorrect footprint 1^3'
    assert np.alltrue(list_footprint[5] == np.array([-1., 0, +1, +1, 0, 0, -1, 0])), 'Incorrect footprint 2^3'

    assert np.alltrue(list_footprint[6] == np.array([+1., -1, -1, -1, +1, +1, +1, -1])), 'Incorrect footprint  1^2^3'

    # three-point interactions with interaction_depth=1
    list_footprint = helper_methods.generate_footprints(list_occluded_features=['empty_set', '1', '2', '3'])
    assert np.alltrue(list_footprint[0] == np.array([+1, -1, 0, 0])), 'Incorrect footprint main effect 1'
    assert np.alltrue(list_footprint[1] == np.array([+1, 0, -1, 0])), 'Incorrect footprint main effect 2'
    assert np.alltrue(list_footprint[2] == np.array([+1, 0, 0, -1])), 'Incorrect footprint main effect 3'

    # four-point interactions with interaction_depth=2
    list_occluded_features = ['empty_set', '1', '2', '3', '4', '1^2', '1^3', '2^3', '2^4', '3^4', '1^4']
    list_footprint = helper_methods.generate_footprints(list_occluded_features=list_occluded_features)
    list_expected_footprints = [np.array([+1, -1, 0, 0, 0,   0, 0, 0, 0, 0, 0]),     # main 1
                                np.array([+1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0]),  # main 2
                                np.array([+1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0]),  # main 3
                                np.array([+1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0]),  # main 4

                                np.array([-1, +1, +1, 0, 0, -1, 0, 0, 0, 0, 0]),  # 1^2
                                np.array([-1, +1, 0, +1, 0, 0, -1, 0, 0, 0, 0]),  # 1^3
                                np.array([-1, 0, +1, +1, 0, 0, 0, -1, 0, 0, 0]),  # 2^3
                                np.array([-1, 0, +1, 0, +1, 0, 0, 0, -1, 0, 0]),  # 2^4
                                np.array([-1, 0, 0, +1, +1, 0, 0, 0, 0, -1, 0]),  # 3^4
                                np.array([-1, +1, 0, 0, +1, 0, 0, 0, 0, 0, -1]),  # 1^4
                                ]
    for ft_generate, ft_expected, features in \
            zip(list_footprint, list_expected_footprints, list_occluded_features[1:]):
        assert np.alltrue(ft_generate == ft_expected), f'Incorrect footprint: {features}'
