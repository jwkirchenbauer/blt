import torch

from bytelatent.norms import fixed_clip_grad_norm_


def test_fixed_clip_grad_norm_clips_fp32_gradients() -> None:
    parameter = torch.nn.Parameter(torch.zeros(4, dtype=torch.float32))
    parameter.grad = torch.tensor([3.0, 4.0, 0.0, 0.0], dtype=torch.float32)

    total_norm = fixed_clip_grad_norm_([parameter], max_norm=1.0, foreach=True)

    torch.testing.assert_close(total_norm.float(), torch.tensor(5.0))
    torch.testing.assert_close(
        parameter.grad.norm(), torch.tensor(1.0), atol=2e-3, rtol=2e-3
    )
