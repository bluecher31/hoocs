"""Provides test functionality applicable to multiple explainer types."""

import numpy as np

from src.hoocs.derived_explainer_targeted import TargetedAttributionsMethod
from src.hoocs.imputers import simple_imputers
from tests import simple_models

from numpy.testing import assert_allclose


def check_additivity(explainer_cls: TargetedAttributionsMethod):
    # TODO: this not the correct way of passing the explainer class
    imputer = simple_imputers.ConstantValueImputer(constant=0)
    explainer = explainer_cls(model_fn=simple_models.additive_syntethic_paper_regression_fn,
                              imputer=imputer, data=np.ones((3, 2, 4)), n_eval=3)

    data = np.array([[1, 1, 1, 1], [1, 2, 3, 4]])
    seg = [7, 1, 2, 8]
    segmentation = np.stack([seg for _ in range(2)])

    explainer.interaction_depth = 1
    dict_attribution_0 = explainer.attribution(data=data, segmentation=segmentation,
                                               target_features={7, 1, 2, 8})
    assert_allclose(dict_attribution_0['7'].reshape(2), [1, 1])
    assert_allclose(dict_attribution_0['1'].reshape(2), [3, 6])
    assert_allclose(dict_attribution_0['2'].reshape(2), [0.06782644, 1.75076841])
    assert_allclose(dict_attribution_0['8'].reshape(2), [-0.5, -32])
