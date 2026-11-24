import math

import pytest

from ramulator._ramulator_test import _ChannelMapperUnderTest


def _mapper(channels, interleave_size):
    return _ChannelMapperUnderTest(
        {
            "impl": "Gem5PortInterleave",
            "interleave_size": interleave_size,
        },
        channels,
        6,
    )


def _addr_for_channel(channels, interleave_size, channel):
    shift = int(math.log2(interleave_size))
    width = int(math.log2(channels))
    low = min(37, interleave_size - 1)
    high = 0x12345
    return (high << (shift + width)) | (channel << shift) | low


def _strip_channel_bits(addr, channels, interleave_size):
    if channels == 1:
        return addr
    shift = int(math.log2(interleave_size))
    width = int(math.log2(channels))
    low = addr & ((1 << shift) - 1)
    high = addr >> (shift + width)
    return (high << shift) | low


@pytest.mark.parametrize("channels", [1, 2, 4, 8])
@pytest.mark.parametrize("interleave_size", [64, 128])
def test_gem5_port_interleave_uses_ingress_id(channels, interleave_size):
    mapper = _mapper(channels, interleave_size)
    for ingress_id in range(channels):
        addr = _addr_for_channel(channels, interleave_size, ingress_id)
        result = mapper.apply(addr, ingress_id=ingress_id, source_id=999)

        assert result["channel"] == ingress_id
        assert result["addr_vec"][0] == ingress_id
        assert result["intra_channel_addr"] == _strip_channel_bits(
            addr,
            channels,
            interleave_size,
        )


@pytest.mark.parametrize(
    ("channels", "interleave_size", "match"),
    [
        (0, 64, "positive channel count"),
        (3, 64, "power-of-two channel count"),
        (4, 0, "power-of-two interleave_size"),
        (4, 96, "power-of-two interleave_size"),
        (4, 32, "at least one Ramulator transaction"),
    ],
)
def test_gem5_port_interleave_rejects_invalid_setup(
    channels,
    interleave_size,
    match,
):
    with pytest.raises(RuntimeError, match=match):
        _mapper(channels, interleave_size)


def test_gem5_port_interleave_trusts_ingress_id():
    mapper = _mapper(4, 64)
    addr = _addr_for_channel(4, 64, 0)
    result = mapper.apply(addr, ingress_id=1)

    assert result["channel"] == 1
    assert result["addr_vec"][0] == 1
    assert result["intra_channel_addr"] == _strip_channel_bits(addr, 4, 64)
