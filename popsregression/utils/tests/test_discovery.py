# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from popsregression.utils.discovery import all_displays, all_estimators, all_functions


def test_all_estimators():
    estimators = all_estimators()
    assert len(estimators) == 2

    estimators = all_estimators(type_filter="regressor")
    assert len(estimators) == 2

    estimators = all_estimators(type_filter="classifier")
    assert len(estimators) == 0

    err_msg = "Parameter type_filter must be"
    with pytest.raises(ValueError, match=err_msg):
        all_estimators(type_filter="xxxx")


def test_all_displays():
    displays = all_displays()
    assert len(displays) == 0


def test_all_functions():
    functions = all_functions()
    assert len(functions) == 8
