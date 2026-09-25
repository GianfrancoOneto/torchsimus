from dataclasses import dataclass
import torch

@dataclass
class ElementPose:
    centers: torch.Tensor   # [N, 3]
    u: torch.Tensor         # [N, 3]
    v: torch.Tensor         # [N, 3]
    n: torch.Tensor         # [N, 3]

def validate_pose(pose: ElementPose, atol: float = 1e-8) -> None:
    ones = torch.ones(pose.centers.shape[0], dtype=pose.centers.dtype, device=pose.centers.device)
    torch.testing.assert_close(torch.linalg.vector_norm(pose.u, dim=-1), ones, atol=atol, rtol=0)
    torch.testing.assert_close(torch.linalg.vector_norm(pose.v, dim=-1), ones, atol=atol, rtol=0)
    torch.testing.assert_close(torch.linalg.vector_norm(pose.n, dim=-1), ones, atol=atol, rtol=0)

    zeros = torch.zeros_like(ones)
    torch.testing.assert_close((pose.u * pose.v).sum(-1), zeros, atol=atol, rtol=0)
    torch.testing.assert_close((pose.u * pose.n).sum(-1), zeros, atol=atol, rtol=0)
    torch.testing.assert_close((pose.v * pose.n).sum(-1), zeros, atol=atol, rtol=0)

    torch.testing.assert_close(torch.cross(pose.u, pose.v, dim=-1), pose.n, atol=atol, rtol=0)
