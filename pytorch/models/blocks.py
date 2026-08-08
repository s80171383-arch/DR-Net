"""DR-Net blocks. TF [B,N,1,C] tensors are represented as PyTorch [B,C,N]."""
import torch
from torch import nn


def gather_neighbour(pc, idx):
    """Gather [B,N,C] with [B,Q,K], returning [B,Q,K,C]."""
    b = torch.arange(pc.shape[0], device=pc.device)[:, None, None]
    return pc[b, idx]


def relative_pos_encoding(xyz, idx):
    neighbours = gather_neighbour(xyz, idx)
    centres = xyz[:, :, None, :].expand_as(neighbours)
    relative = centres - neighbours
    distance = torch.sqrt(torch.sum(relative.square(), dim=-1, keepdim=True))
    return torch.cat((distance, relative, centres, neighbours), dim=-1)


class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, activation=True):
        layers = [nn.Conv1d(cin, cout, 1), nn.BatchNorm1d(cout, momentum=0.01, eps=1e-6)]
        if activation:
            layers.append(nn.LeakyReLU(inplace=True))
        super().__init__(*layers)


class AttentionPooling(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.attention = nn.Linear(cin, cin, bias=False)
        self.out = ConvBNAct(cin, cout)

    def forward(self, features):
        scores = torch.softmax(self.attention(features), dim=2)
        aggregated = (features * scores).sum(dim=2).transpose(1, 2)
        return self.out(aggregated)


class BuildingBlock(nn.Module):
    """Faithful LFA: idx12 is accepted but deliberately unused by the TF graph."""
    def __init__(self, din, dout, compatibility_mode=True):
        super().__init__()
        if not compatibility_mode:
            raise NotImplementedError("corrected behavior is intentionally not implemented")
        self.pos0, self.pos1 = ConvBNAct(10, din), ConvBNAct(10, din)
        self.pool0 = AttentionPooling(2 * din, dout // 4)
        self.pool1 = AttentionPooling(2 * din, dout // 4)
        self.pos_final = ConvBNAct(din, dout // 2)
        self.pool_final = AttentionPooling(dout, dout)

    def forward(self, xyz, feature, idx16, idx8, idx12):
        del idx12  # compatibility invariant: K=12 branch never enters aggregation
        p0 = self.pos0(relative_pos_encoding(xyz, idx16).permute(0, 3, 1, 2).flatten(2)).view(feature.shape[0], -1, xyz.shape[1], idx16.shape[-1]).permute(0,2,3,1)
        p1 = self.pos1(relative_pos_encoding(xyz, idx8).permute(0, 3, 1, 2).flatten(2)).view(feature.shape[0], -1, xyz.shape[1], idx8.shape[-1]).permute(0,2,3,1)
        f = feature.transpose(1, 2)
        a0 = self.pool0(torch.cat((gather_neighbour(f, idx16), p0), -1))
        a1 = self.pool1(torch.cat((gather_neighbour(f, idx8), p1), -1))
        added = torch.cat((a0, a1), 1)
        pf = self.pos_final(p0.permute(0,3,1,2).flatten(2)).view(feature.shape[0], -1, xyz.shape[1], idx16.shape[-1]).permute(0,2,3,1)
        return self.pool_final(torch.cat((gather_neighbour(added.transpose(1,2), idx16), pf), -1))


class DilatedResidualBlock(nn.Module):
    def __init__(self, cin, dout, compatibility_mode=True):
        super().__init__()
        self.mlp1 = ConvBNAct(cin, dout // 2)
        self.block = BuildingBlock(dout // 2, dout, compatibility_mode)
        self.mlp2 = ConvBNAct(dout, dout * 2, activation=False)
        self.shortcut = ConvBNAct(cin, dout * 2, activation=False)

    def forward(self, feature, xyz, idx16, idx8, idx12):
        return torch.nn.functional.leaky_relu(self.mlp2(self.block(xyz, self.mlp1(feature), idx16, idx8, idx12)) + self.shortcut(feature))


def random_sample(feature, pool_idx):
    return gather_neighbour(feature.transpose(1, 2), pool_idx).amax(dim=2).transpose(1, 2)
