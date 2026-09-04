import pytest
import torch

from tools.artifact import (
    decode_fp8_block_scaled_words,
    dequantize_fp8_block_scaled,
    encode_fp8_block_scaled,
    encoded_size,
    fp8_block_scale_geometry,
)


def test_block_fp8_roundtrip_and_dequantization() -> None:
    codes = torch.tensor([0x38, 0xB8, 0x30, 0xB0], dtype=torch.uint8).repeat(128, 64)
    scales = torch.tensor([[0.25, 0.5]], dtype=torch.float32)
    payload = encode_fp8_block_scaled(codes, scales, codes.shape)
    geometry = fp8_block_scale_geometry("FP8_E4M3FN_BLOCK128_F32S", codes.shape)
    assert len(payload) == geometry.payload_bytes
    assert len(payload) == encoded_size(
        "blockscale-k128-m128-v1", "FP8_E4M3FN_BLOCK128_F32S", codes.shape
    )
    actual_codes, actual_scales = decode_fp8_block_scaled_words(payload, codes.shape)
    assert torch.equal(actual_codes, codes)
    assert torch.equal(actual_scales, scales)
    expected = codes.view(torch.float8_e4m3fn).float()
    expected[:, :128] *= 0.25
    expected[:, 128:] *= 0.5
    assert torch.equal(dequantize_fp8_block_scaled(payload, codes.shape), expected)


def test_block_fp8_rejects_bad_geometry_and_scales() -> None:
    with pytest.raises(ValueError, match="divisible by 128"):
        encoded_size("blockscale-k128-m128-v1", "FP8_E4M3FN_BLOCK128_F32S", (128, 129))
    codes = torch.zeros((128, 128), dtype=torch.uint8)
    with pytest.raises(ValueError, match="finite and positive"):
        encode_fp8_block_scaled(codes, torch.zeros((1, 1), dtype=torch.float32), codes.shape)
