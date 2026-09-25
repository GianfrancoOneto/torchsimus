import torch

def local_geometry(scatterers, pose):
    d = scatterers[:, None, :] - pose.centers[None, :, :]
    r = torch.linalg.vector_norm(d, dim=-1)
    x = (d * pose.u[None]).sum(-1)
    y = (d * pose.v[None]).sum(-1)
    z = (d * pose.n[None]).sum(-1)
    return x, y, z, r
