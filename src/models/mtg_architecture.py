from torchaudio.transforms import MelSpectrogram
from torch import nn
import torch


class MTGConvBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 kernel_size, stride, padding, max_pool_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU()
        self.maxpool = nn.MaxPool2d(max_pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.maxpool(x)
        return x


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        # self.melspec = MelSpectrogram(
        #     n_mels=64, sample_rate=sample_rate
        # )

        self.conv1 = MTGConvBlock(1, 64, 3, 1, 'same', 2)
        self.conv2 = MTGConvBlock(64, 128, 3, 1, 'same', 2)
        self.conv3 = MTGConvBlock(128, 128, 3, 1, 'same', 2)
        self.conv4 = MTGConvBlock(128, 128, 3, 1, 'same', 2)
        self.conv5 = MTGConvBlock(128, 64, 3, 1, 'same', 2)

    def forward(self, x: torch.Tensor):
        assert x.ndim == 3, "Expected a batch of audio samples shape (batch, channels, samples)"

        # x = self.melspec(x)

        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # pool over the time dimension
        # squeeze the (t, f) dimensions
        x = x.mean(dim=-1)
        x = x.squeeze(-1).squeeze(-1)  # (batch, 512)

        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class PrototypicalNet(nn.Module):

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, support: dict, query: dict):
        """
        Forward pass through the protonet.

        Args:
            support (dict): A dictionary containing the support set.
                The support set dict must contain the following keys:
                    - audio: A tensor of shape (n_support, n_channels, n_samples)
                    - label: A tensor of shape (n_support) with label indices
                    - classlist: A tensor of shape (n_classes) containing the list of classes in this episode
            query (dict): A dictionary containing the query set.
                The query set dict must contain the following keys:
                    - audio: A tensor of shape (n_query, n_channels, n_samples)

        Returns:
            logits (torch.Tensor): A tensor of shape (n_query, n_classes) containing the logits

        After the forward pass, the support dict is updated with the following keys:
            - embeddings: A tensor of shape (n_support, n_features) containing the embeddings
            - prototypes: A tensor of shape (n_classes, n_features) containing the prototypes

        The query dict is updated with
            - embeddings: A tensor of shape (n_query, n_features) containing the embeddings

        """
        # compute the embeddings for the support and query sets
        support["embeddings"] = self.backbone(support["audio"])
        query["embeddings"] = self.backbone(query["audio"])

        # group the support embeddings by class
        support_embeddings = []
        for idx in range(len(support["classlist"])):
            embeddings = support["embeddings"][support["target"] == idx]
            support_embeddings.append(embeddings)
        support_embeddings = torch.stack(support_embeddings)

        # compute the prototypes for each class
        prototypes = support_embeddings.mean(dim=1)
        support["prototypes"] = prototypes

        print(support_embeddings.shape)
        print(prototypes.shape)
        print(query["embeddings"].shape)
        # compute the distances between each query and prototype
        distances = torch.cdist(
            query["embeddings"].unsqueeze(0),
            prototypes.unsqueeze(0),
            p=2
        ).squeeze(0)

        # square the distances to get the sq euclidean distance
        distances = distances ** 2
        logits = -distances

        # return the logits
        return logits


if __name__ == '__main__':
    backbone = Backbone()

    print(backbone)
