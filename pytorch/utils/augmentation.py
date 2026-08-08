import torch

ROTATION_CONSTANT = 3.14592653


def rotation_matrix_y(theta, *, device=None, dtype=torch.float32):
    t = torch.as_tensor(theta, device=device, dtype=dtype)
    c, s = torch.cos(t), torch.sin(t)
    return torch.stack((torch.stack((c, t*0, -s)), torch.stack((t*0, t*0+1, t*0)), torch.stack((s, t*0, c))))


def augment_features(data, option=None, theta=None, generator=None):
    xyz = data[..., :3].clone()
    option = int(torch.randint(3, (), generator=generator)) if option is None else option
    if option == 0:
        xyz[..., 1].neg_()
    elif option == 1:
        theta = float(2 * ROTATION_CONSTANT * torch.rand((), generator=generator)) if theta is None else theta
        xyz = xyz.reshape(-1, 3) @ rotation_matrix_y(theta, device=xyz.device, dtype=xyz.dtype)
        xyz = xyz.reshape_as(data[..., :3])
    elif option == 2:
        # Exactly one [N,3] draw, shared across every cloud in the batch.
        jitter = (0.01 * torch.randn((data.shape[1], 3), generator=generator, device=data.device, dtype=data.dtype)).clamp(-0.05, 0.05)
        xyz = xyz + jitter[None]
    out = torch.cat((xyz, data[..., 3:]), -1)
    return out


def prepare_training_batch(features, anno_xyz, hierarchy, training, **augment_kwargs):
    if not training:
        return features, anno_xyz, hierarchy
    augmented = augment_features(features, **augment_kwargs)
    doubled = [{k: torch.cat((v, v), 0) for k, v in level.items()} for level in hierarchy]
    return torch.cat((features, augmented), 0), torch.cat((anno_xyz, anno_xyz), 0), doubled
