import json

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(left, right, output, count: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < count
    value = tl.load(left + offsets, mask=mask) + tl.load(right + offsets, mask=mask)
    tl.store(output + offsets, value, mask=mask)


count = 257
left = torch.randn(count, device="cuda")
right = torch.randn(count, device="cuda")
output = torch.empty_like(left)
add_kernel[(triton.cdiv(count, 256),)](left, right, output, count, 256)
expected = left + right
error = float((output - expected).abs().max())
print(
    json.dumps(
        {
            "compiled": True,
            "correct": torch.allclose(output, expected, atol=1e-4, rtol=1e-4),
            "max_abs_error": error,
            "shape": list(output.shape),
        }
    )
)
