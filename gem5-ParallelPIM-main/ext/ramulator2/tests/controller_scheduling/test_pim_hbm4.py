import pytest

from tests.controller_scheduling.pim_cases import CASES

pytestmark = pytest.mark.controller_scheduling


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.__name__)
def test_pim_hbm4(case):
    """gem5 PIM sequencer on HBM4_PIM."""
    case("HBM4")
