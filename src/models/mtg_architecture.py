from common_architecture import *
from src.data.make_dataset import MTGJamendo, EpisodeDataset


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
        self.conv5 = MTGConvBlock(128, 64, 3, 1, 'same', 4)

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


if __name__ == '__main__':
    dataset = MTGJamendo(False,
                         'autotagging_moodtheme',
                         'melspecs',
                         'mtg-fast',
                         'D:/magisterka-dane',
                         True,
                         True,
                         '../data/mtg_jamendo_dataset/data/autotagging_moodtheme.tsv',
                         '../data/mtg_jamendo_dataset/data/tags/moodtheme.txt',
                         None)

    episodes = EpisodeDataset(
        dataset,
        n_way=5,
        n_support=5,
        n_query=20,
        n_episodes=100,
    )

    support, query = episodes[0]
    episodes.print_episode(support, query)

    backbone = Backbone()
    protonet = PrototypicalNet(backbone)

    logits = protonet(support, query)
    print(f"got logits with shape {logits.shape}")
