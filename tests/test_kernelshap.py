"""
Test routines for Shapley Values including interaction indices.
"""
import numpy as np
from numpy.testing import assert_allclose

from hoocs import kernelshap
from hoocs.imputers import simple_imputers
from tests import simple_models

from typing import Dict, Set


def check_efficiency(dict_attributions: Dict, target_features: Set[int]):
    for i in range(len(dict_attributions['true_prediction'])):
        tmp = float(dict_attributions['mean_prediction'][i])
        for feature in target_features:
            tmp += dict_attributions[str(feature)][i]
        assert np.allclose(tmp, dict_attributions['true_prediction'][i], atol=0.05)


class TestKernelSHAP:
    def test_attribution_additive_regression(self):
        """Test additive attributions using the maximum_interaction capability."""
        # impute_zeros_fn = partial(simple_imputers.impute_constant_fn, constant=0)
        imputer = simple_imputers.ConstantValueImputer(constant=0)
        explainer = kernelshap.KernelSHAP(model_fn=simple_models.additive_syntethic_paper_regression_fn,
                                          imputer=imputer, data=np.ones((3, 2, 4)), n_eval=15)

        data = np.array([[1, 1, 1, 1], [1, 2, 3, 4]])
        seg = [7, 1, 2, 8]
        segmentation = np.stack([seg for _ in range(2)])
        target_features = set(segmentation.flatten())

        dict_attributions = explainer.attribution(data=data, segmentation=segmentation, target_features=target_features)
        dict_target_attributions = {'7': [1, 1], '1': [3, 6], '2': [0.06782644, 1.75076841], '8': [-0.5, -32]}
        for key in dict_target_attributions:
            assert_allclose(dict_attributions[key][:, 0], dict_target_attributions[key], atol=0.5,
                            err_msg=f'Incorrect attributions for key = {key}\n'
                                    f'x = attribution, y = target')

        check_efficiency(dict_attributions, target_features)

    def test_two_point(self):
        f = simple_models.two_point_interaction_regression_fn
        imputer = simple_imputers.ConstantValueImputer(constant=0)

        data = np.array([[1, 1, 1, 1], [1, 2, 3, 4]])
        # f([a, b])     =   [2.5   1.]
        # f([a, b]\1)   =   [0.5  -1.]
        # f([a, b]\2)   =   [2.    2.]
        # f([a, b]\3)   =   [3.    4.]
        # f([a, b]\4)   =   [2.5.  1.]
        # f([a, b]\12)  =   [0.    0.]
        # f([a, b]\13)  =   [1.    2.]
        # f([a, b]\23)  =   [2.    2.]
        # f([a, b]\123) =   [0.    0.]
        # f([a, b]\1234)=   [0.    0.]

        dict_target_attributions = {'1': [2, 2],
                                    '2': [0.75, 0.5],
                                    '3': [-0.25, -1.5],
                                    '4': [0, 0]}

        seg = [1, 2, 3, 4]
        segmentation = np.stack([seg for _ in range(2)])

        for subsample in [True, False]:
            explainer = kernelshap.KernelSHAP(model_fn=f, imputer=imputer,
                                              data=np.ones((3, 2, 4)), n_eval=5_000, subsample_coalitions=subsample)
            dict_attributions = explainer.attribution(data=data, segmentation=segmentation,
                                                      target_features={1, 2, 3, 4})
            for key in dict_target_attributions:
                assert_allclose(dict_attributions[key][:, 0], dict_target_attributions[key], atol=0.1,
                                err_msg=f'subsample_coalitions = {subsample}'
                                        f'Incorrect attributions for key = {key}\n'
                                        f'x = attribution, y = target')

        for subsample in [True, False]:
            # TODO: this test does not converge to the correct mean, atol -> 0 for n_eval -> 20_000
            explainer = kernelshap.KernelSHAP(model_fn=f, imputer=imputer,
                                              data=np.ones((3, 2, 4)), n_eval=5_000, subsample_coalitions=subsample,
                                              cardinality_coalitions=[-3, -2])
            dict_attributions = explainer.attribution(data=data, segmentation=segmentation,
                                                      target_features={1, 2, 3, 4})
            for key in dict_target_attributions:
                assert_allclose(dict_attributions[key][:, 0], dict_target_attributions[key], atol=0.5,
                                err_msg=f'subsample_coalitions = {subsample}\n'
                                        f'Incorrect attributions for key = {key}\n'
                                        f'x = attribution, y = target\n')

    # @pytest.mark.skip(reason="no way of currently testing this")
    def test_1point_approximation(self):
        """
        Using only coalitions with |S| = 1, this correspond to PredDiff up to rescaling.
        """
        imputer = simple_imputers.ConstantValueImputer(constant=0)
        f = simple_models.four_point_regression_fn
        a = np.array([1, 2, -1, 0.5])         # f(a) = 7.91917545
        b = np.array([2, -1, 1.5, -1])        # f(b) = -12.41838102
        data = np.stack([a, b])
        # f(a\0) = 1,               f(b\0) = 1
        # m^a_0 = 6.91917545        m^b_0 = -13.41838102
        # f(a\1) = 2,               f(b\1) = 3
        # m^a_0 = 5.91917545        m^b_0 = -15.41838102
        # f(a\2) = 11.68649864,     f(b\2) = -0.80770025
        # m^a_0 = -3.76732319       m^b_0 = -11.61068077
        # f(a\3) = 7.44187301,     f(b\3) = -35.08212342
        # m^a_0 = 0.477302439        m^b_0 = 22.663742400000004

        # rescale PredDiff attributions to obtain efficiency
        # sum of mbar values: 9.548330149/-17.783700409999994
        # f(a)/f(b) - f(empty) = 6.91917545/-13.41838102
        # rescale factor, i.e.,
        # a: 6.91917545/9.548330149 =  0.7246476967205253
        # b: -13.41838102/-17.783700409999994 = 0.7545325613141053

        scale_a = 0.7246476967205253
        scale_b = 0.7545325613141053
        dict_target_attributions = {'1': np.array([[scale_a * 6.91917545], [scale_b * -13.41838102]]),
                                    '2': np.array([[scale_a * 5.91917545], [scale_b * -15.41838102]]),
                                    '3': np.array([[scale_a * -3.76732319], [scale_b * -11.61068077]]),
                                    '4': np.array([[scale_a * 0.477302439], [scale_b * 22.663742400000004]]),
                                    'true_prediction': f(data),
                                    'mean_prediction': f(np.zeros_like(data))}


        # calculate all occluded predictions
        # f([a, b]) = [ 7.91917545, -12.41838102]

        # f([a, b]\1) = [1, 1]
        # f([a, b]\2) = [2, 3]
        # f([a, b]\3) = [11.68649864, -0.80770025]
        # f([a, b]\4) = [ 7.44187301, -35.08212342]

        # f([a, b]\12) = [1, 1]
        # f([a, b]\13) = [1, 1]
        # f([a, b]\14) = [1, 1]
        # f([a, b]\23) = [2, 3]
        # f([a, b]\24) = [2, 3]
        # f([a, b]\34) = [10.3890561 , -5.86466472]

        # f([a, b]\123) = [1, 1]
        # f([a, b]\124) = [1, 1]
        # f([a, b]\134) = [1, 1]
        # f([a, b]\234) = [2, 3]

        # f([a, b]\1234) = [1, 1]
        # imp_data = data.copy()
        # # imp_data[:, 3] = 0
        # print(imp_data)
        # print('result: \n', np.squeeze(f(imp_data)))

        # INFO: ridge_parameter is manually tuned to match KernelSHAP onto PredDiff attributions.
        explainer = kernelshap.KernelSHAP(model_fn=f, cardinality_coalitions=[-1],
                                          imputer=imputer, data=np.ones((3, 2, 4)), n_eval=2000,
                                          subsample_coalitions=False, ridge_parameter=0.08)

        segmentation = np.stack([np.arange(1, 5) for _ in range(2)])
        dict_attributions = explainer.attribution(data=data, segmentation=segmentation,
                                                  target_features=set(segmentation.flatten()))

        check_efficiency(dict_attributions=dict_target_attributions, target_features=set(segmentation.flatten()))
        check_efficiency(dict_attributions, set(segmentation.flatten()))

        for key in dict_target_attributions:
            predicted = dict_attributions[key][:, 0]
            scale = 10.1246054/6.79731509           # need to rescale for each sample individually
            expected = dict_target_attributions[key][:, 0]
            print(f'predicted: {dict_attributions[key][:, 0]}\n'
                  f'expected: {dict_target_attributions[key][:, 0]}\n'
                  f'rescaled predicted: {predicted*scale}\n'
                  )

        for key in dict_target_attributions:
            assert_allclose(dict_attributions[key][:, 0], dict_target_attributions[key][:, 0], atol=0.5,
                            err_msg=f'Incorrect attributions for key = {key}\n'
                                    f'x = attribution, y = target')

    # TODO: Add test with superpixel features, i.e. calculate attribution jointly for multiples features

    # TODO: add KernelSHAP with S=1 (corresponding to PredDiff with completeness rescaling),
    #  thereby also complex function are easily computable
