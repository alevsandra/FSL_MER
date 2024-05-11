from common_architecture import *
from src.data.make_dataset import *


class ConvBlock(nn.Module):
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
    def __init__(self):
        super().__init__()

        self.conv1 = ConvBlock(1, 32, 3, 1, 'same', 8, 2)
        self.conv2 = ConvBlock(32, 64, 3, 1, 'same', 16, 2)
        self.conv3 = ConvBlock(64, 128, 3, 1, 'same', 32, 2)
        self.conv4 = ConvBlock(128, 256, 3, 1, 'same', 64, 2)
        self.conv5 = ConvBlock(256, 512, 1, 1, 'same', 128, 2)

    def forward(self, x: torch.Tensor):
        assert x.ndim == 3, "Expected a batch of audio samples shape (batch, channels, samples)"

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


class BackboneMTG(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = MTGConvBlock(1, 64, 3, 1, 'same', 2)
        self.conv2 = MTGConvBlock(64, 128, 3, 1, 'same', 2)
        self.conv3 = MTGConvBlock(128, 128, 3, 1, 'same', 2)
        self.conv4 = MTGConvBlock(128, 128, 3, 1, 'same', 2)
        self.conv5 = MTGConvBlock(128, 64, 3, 1, 'same', 4)

    def forward(self, x: torch.Tensor):
        assert x.ndim == 3, "Expected a batch of audio samples shape (batch, channels, samples)"

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
    # only non-empty classes
    dataset = PMEmo(False, ['power', 'surprise', 'tension', 'sadness', 'tenderness', 'transcendence'])

    # dataset = MTGJamendo(False,
    #                      'D:/magisterka-dane',
    #                      '../data/mtg_jamendo_dataset/data/autotagging_moodtheme.tsv',
    #                      '../data/mtg_jamendo_dataset/data/tags/moodtheme.txt',
    #                      ['ambiental', 'background', 'ballad', 'calm', 'cool', 'dark', 'deep', 'dramatic', 'dream',
    #                       'emotional', 'energetic', 'epic', 'fast', 'fun', 'funny', 'groovy', 'happy', 'heavy',
    #                       'hopeful', 'horror', 'inspiring', 'love', 'meditative', 'melancholic', 'mellow', 'melodic',
    #                       'motivational', 'nature', 'party', 'positive', 'powerful', 'relaxing', 'retro', 'romantic',
    #                       'sad'])

    # dataset = JointDataset(False,
    #                        ['joy', 'power', 'surprise', 'anger', 'tension', 'fear', 'sadness', 'bitterness', 'peace',
    #                         'tenderness', 'transcendence'])

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

    # backbone = BackboneMTG()

    # for joint and each sub dataset
    backbone = Backbone()

    protonet = PrototypicalNet(backbone)

    logits = protonet(support, query)
    print(f"got logits with shape {logits.shape}")
