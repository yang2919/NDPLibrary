import pytest

from tests.device_timings.pim_cases import CASES

pytestmark = pytest.mark.device_timings


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.__name__)
def test_pim_gddr7(case):
    """Earliest-legal-cycle tests of gem5's PIM commands on GDDR7_PIM."""
    case("GDDR7")
