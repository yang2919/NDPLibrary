import pytest

from tests.device_timings.pim_cases import CASES

pytestmark = pytest.mark.device_timings


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.__name__)
def test_pim_hbm4(case):
    """Earliest-legal-cycle tests of gem5's PIM commands on HBM4_PIM."""
    case("HBM4")
