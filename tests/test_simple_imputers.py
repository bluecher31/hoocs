import numpy as np

from hoocs.imputers import simple_imputers
from hoocs import helper_base


class TestConstantValueImputer():
    def test_impute_zero(self):
        imputer = simple_imputers.ConstantValueImputer(constant=0)
        impute_zeros_fn = imputer.impute
        data = np.random.randn(3 * 4 * 5).reshape((3, 4, 5))
        helper_base.partial_check_impute_fn(impute_fn=impute_zeros_fn, data=data)

        zeros = impute_zeros_fn(data=data, segmentation_coalitions=-np.ones_like(data[None]), n_imputations=2)
        assert zeros.sum() == 0, 'Impute all zeros is not 0.'


def test_impute_identity_x():
    imputer = simple_imputers.IdentityImputer()
    helper_base.partial_check_impute_fn(impute_fn=imputer.impute,
                                        data=np.random.randn(3 * 4 * 5).reshape((3, 4, 5)))


def test_trainset_imputer():
    imputer = simple_imputers.TrainSetImputer(train_data=np.random.randn(3 * 4 * 5).reshape((3, 4, 5)))
    helper_base.partial_check_impute_fn(impute_fn=imputer.impute,
                                        data=np.random.randn(3 * 4 * 5).reshape((3, 4, 5)))


def test_gaussian_variable_imputer():
    mu_data = np.random.randn(5)
    cov_data = np.eye(5)
    imputer = simple_imputers.GaussianNoiseImputer(mu_data, cov_data)
    helper_base.partial_check_impute_fn(impute_fn=imputer.impute,
                                        data=np.random.randn(3 * 4 * 5).reshape((3, 4, 5)))
