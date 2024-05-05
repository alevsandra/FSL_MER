from torchaudio.transforms import MelSpectrogram
from common_architecture import *
from src.data.make_dataset import PMEmo, EpisodeDataset


class PMEmoConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, num_groups, max_pool_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.gn = nn.GroupNorm(num_groups, out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(max_pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class Backbone(nn.Module):
    def __init__(self, sample_rate: int):
        super().__init__()
        self.melspec = MelSpectrogram(n_mels=32, sample_rate=sample_rate)

        self.conv1 = PMEmoConvBlock(1, 32, 3, 1, 'same', 8, 2)
        self.conv2 = PMEmoConvBlock(32, 64, 3, 1, 'same', 16, 2)
        self.conv3 = PMEmoConvBlock(64, 128, 3, 1, 'same', 32, 2)
        self.conv4 = PMEmoConvBlock(128, 256, 3, 1, 'same', 64, 2)
        self.conv5 = PMEmoConvBlock(256, 512, 1, 1, 'same', 128, 2)

    def forward(self, x: torch.Tensor):
        assert x.ndim == 3, "Expected a batch of audio samples shape (batch, channels, samples)"

        x = self.melspec(x)

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
    # only non-empty classes
    dataset = PMEmo(False, ['power', 'surprise', 'tension', 'sadness', 'tenderness', 'transcendence'])

    for key, item in dataset.class_to_indices.items():
        print(key, len(item))

    episodes = EpisodeDataset(
        dataset,
        n_way=5,
        n_support=5,
        n_query=20,
        n_episodes=100,
    )

    support, query = episodes[0]
    episodes.print_episode(support, query)

    sr = 44100
    backbone = Backbone(sr)
    protonet = PrototypicalNet(backbone)

    logits = protonet(support, query)
    print(f"got logits with shape {logits.shape}")
