import torch
from torch import nn
from .blocks import ConvBNAct, DilatedResidualBlock, random_sample
from utils.interpolation import semantic_query
from utils.knn import knn_search
from utils.augmentation import prepare_training_batch


class DRNet(nn.Module):
    """Faithful DALES DR-Net core (TF [B,N,1,C] is [B,C,N] internally)."""
    def __init__(self, input_dim=3, num_classes=8, d_out=(16,64,128,256,512), compatibility_mode=True):
        super().__init__()
        if not compatibility_mode:
            raise NotImplementedError("Only faithful compatibility mode exists in stage 1")
        if len(d_out) != 5 or num_classes != 8:
            raise ValueError("DALES compatibility mode requires five layers and 8 logits")
        self.compatibility_mode = True
        self.fc0 = ConvBNAct(input_dim, 8)
        channels = [8] + [2*d for d in d_out[:-1]]
        self.encoders = nn.ModuleList(DilatedResidualBlock(c, d, True) for c, d in zip(channels, d_out))
        # TF conv2d_transpose, kernel/stride 1x1: no geometric upsampling.
        interp_channels = [2*d for d in d_out]
        reversed_channels = list(reversed(interp_channels))
        decoder_inputs = [2 * reversed_channels[0]] + [a + b for a, b in zip(reversed_channels[:-1], reversed_channels[1:])]
        self.decoders = nn.ModuleList(nn.Sequential(nn.ConvTranspose2d(cin, cout, 1, 1), nn.BatchNorm2d(cout, eps=1e-6, momentum=0.01), nn.LeakyReLU()) for cin, cout in zip(decoder_inputs, reversed_channels))
        total = sum(reversed(interp_channels))
        self.head = nn.Sequential(ConvBNAct(total,256), ConvBNAct(256,128), ConvBNAct(128,64), ConvBNAct(64,64), ConvBNAct(64,32), nn.Conv1d(32,num_classes,1))

    def forward(self, features, anno_xyz, hierarchy, augment_kwargs=None):
        features, anno_xyz, hierarchy = prepare_training_batch(features, anno_xyz, hierarchy, self.training, **(augment_kwargs or {}))
        feature = self.fc0(features.transpose(1,2))
        interpolated = []
        for encoder, level in zip(self.encoders, hierarchy):
            encoded = encoder(feature, level["xyz"], level["neigh_idx"], level["neigh_idx_1"], level["neigh_idx_2"])
            query_idx = knn_search(level["xyz"], anno_xyz, 3)
            # Distances are intentionally irrelevant: fixed arithmetic mean.
            interpolated.append(semantic_query(encoded, query_idx))
            feature = random_sample(encoded, level["sub_idx"])
        feature = interpolated[-1]
        decoded = []
        for skip, decoder in zip(reversed(interpolated), self.decoders):
            feature = decoder(torch.cat((skip, feature), 1).unsqueeze(2)).squeeze(2)
            decoded.append(feature)
        return self.head(torch.cat(decoded, 1)).transpose(1,2)
