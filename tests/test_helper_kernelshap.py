import numpy as np
import pytest

from hoocs import helper_kernelshap

from numpy.testing import assert_allclose

from typing import List, Set


def difference_between_coalitions(coalitions1: List[Set[int]], coalitions2: List[Set[int]]):
    """Checks whether all elements of both list are equal"""
    a = np.setdiff1d(coalitions1, coalitions2)
    b = np.setdiff1d(coalitions2, coalitions1)
    total_difference = np.concatenate([a, b])
    return total_difference


def test_power_set_coalitions():
    coalitions = helper_kernelshap.power_set_coalitions(features={1, 2})
    expected_coalitions = [{1}, {2}]
    diff = difference_between_coalitions(coalitions, expected_coalitions)
    assert len(diff) == 0, f'Found difference between constructed and expected list of coalitions: {diff}'

    coalitions = helper_kernelshap.power_set_coalitions(features={1, 2, 3}, cardinality_coalitions=None)
    expected_coalitions = [{1}, {2}, {3}, {1, 2}, {1, 3}, {2, 3}]
    diff = difference_between_coalitions(coalitions, expected_coalitions)
    assert len(diff) == 0, f'Found difference between constructed and expected list of coalitions: {diff}'

    coalitions = helper_kernelshap.power_set_coalitions(features={1, 2, 3, 4}, cardinality_coalitions=[-1])
    expected_coalitions = [{1, 2, 3}, {1, 2, 4}, {1, 3, 4}, {2, 3, 4}]
    diff = difference_between_coalitions(coalitions, expected_coalitions)
    assert len(diff) == 0, f'Found difference between constructed and expected list of coalitions: {diff}'

    coalitions = helper_kernelshap.power_set_coalitions(features={1, 2, 3, 4}, cardinality_coalitions=[-1, -3])
    expected_coalitions = [{1, 2, 3}, {1, 2, 4}, {1, 3, 4}, {2, 3, 4}, {1}, {2}, {3}, {4}]
    diff = difference_between_coalitions(coalitions, expected_coalitions)
    assert len(diff) == 0, f'Found difference between constructed and expected list of coalitions: {diff}'

    coalitions = helper_kernelshap.power_set_coalitions(features={1, 2, 3, 4, 5, 6}, cardinality_coalitions=[1, 2, -1])
    expected_coalitions = [{1}, {2}, {3}, {4}, {5}, {6},
                           {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {2, 3}, {2, 4}, {2, 5}, {2, 6},
                           {3, 4}, {3, 5}, {3, 6}, {4, 5}, {4, 6}, {5, 6},
                           {2, 3, 4, 5, 6}, {1, 3, 4, 5, 6}, {1, 2, 4, 5, 6}, {1, 2, 3, 5, 6}, {1, 2, 3, 4, 6},
                           {1, 2, 3, 4, 5}]
    diff = difference_between_coalitions(coalitions, expected_coalitions)
    assert len(diff) == 0, f'Found difference between constructed and expected list of coalitions: {diff}'


def test_random_coalitions():
    features = {1, 2, 3}
    coalitions = helper_kernelshap.random_coalitions(features=features, n_coalitions=50, cardinality_coalitions=[1, 2])
    expected_coalitions = [{1}, {2}, {3}, {1, 2}, {1, 3}, {2, 3}]
    diff = difference_between_coalitions(coalitions, expected_coalitions)
    assert len(diff) == 0, f'Found difference between constructed and expected list of coalitions: {diff}'
    assert 50 == len(coalitions), 'Incorrect number of coalitions'

    features = {1, 2, 3}
    coalitions = helper_kernelshap.random_coalitions(features=features, n_coalitions=2, cardinality_coalitions=None)
    assert len(coalitions) == 2, 'Expected two coalitions'

    features = {1, 2, 3, 4, 5, 6}
    coalitions = helper_kernelshap.random_coalitions(features=features, n_coalitions=5, cardinality_coalitions=[1])
    assert len(coalitions) == 5, 'Expected five coalitions'
    assert np.alltrue([len(c) == 1 for c in coalitions]), 'Expected coalitions of size 1'

    coalitions = helper_kernelshap.random_coalitions(features=features, n_coalitions=10, cardinality_coalitions=[1, 3])
    assert len(coalitions) == 10, 'Expected five coalitions'
    assert np.alltrue([len(c) == 1 or len(c) == 3 for c in coalitions]), 'Expected coalitions of size 1'

    coalitions = helper_kernelshap.random_coalitions(features=features, n_coalitions=5, cardinality_coalitions=[-1])
    assert np.alltrue([len(c) == 5 for c in coalitions]), 'Expected coalitions of size 6-1 = 5'


def test_convert_coalitions_to_segmentation():
    segmentation = np.stack(np.arange(1, 5) for _ in range(2))
    list_coalitions = [{1}, {2}]
    segmentation_coalitions = helper_kernelshap.convert_coalitions_to_segmentation(segmentation=segmentation,
                                                                                   list_coalitions=list_coalitions)
    assert_allclose(np.array(
        [[[[1, -2, -3, -4],
           [1, -2, -3, -4]],
          [[-1, 2, -3, -4],
           [-1, 2, -3, -4]]]]),
        segmentation_coalitions
    )

    list_coalitions = [{1, 2}, {2, 3, 4}]
    segmentation_coalitions = helper_kernelshap.convert_coalitions_to_segmentation(segmentation=segmentation,
                                                                                   list_coalitions=list_coalitions)
    assert_allclose(np.array(
        [[[[1, 2, -3, -4],
           [1, 2, -3, -4]],
          [[-1, 2, 3, 4],
           [-1, 2, 3, 4]]]]),
        segmentation_coalitions
    )
