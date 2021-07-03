from torch import nn

img_channels = 4  # stacking 4 images together


class DQN_Conv_Block(nn.Module):
    def __init__(self):
        super(DQN_Conv_Block, self).__init__()
        self.conv_layer = nn.Sequential(

            # TODO: Change to kernel sizes of 3 and no strides?
            # TODO: Fix dimensions of convolution layers
            # Conv Block (Feature Extraction)
            nn.Conv2d(in_channels=img_channels, out_channels=32, kernel_size=8, stride=4),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_layer(x)


class DQN(nn.Module):
    """DQN Network for the Dino Run Game"""

    def __init__(self, ACTIONS):
        super(DQN, self).__init__()
        self.conv_layer = DQN_Conv_Block()

        self.fc_layer = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, ACTIONS)
        )

    def forward(self, x):
        x = x.float() / 255

        # conv layer
        features = self.conv_layer(x)

        # flatten
        features = features.view(features.size(0), -1)
        # print(f'features size: {features.size()}')

        # fully connected
        q_vals = self.fc_layer(features)

        return q_vals


class DuelingDQN(nn.Module):
    """Basic Dueling DQN Network"""

    def __init__(self, ACTIONS):
        super(DuelingDQN, self).__init__()

        # Conv Block (Feature Extraction)
        self.conv_layer = DQN_Conv_Block()

        self.value_stream = nn.Sequential(
            nn.Linear(in_features=4096, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(in_features=4096, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=ACTIONS)
        )

    def forward(self, x):
        x = x.float() / 255
        features = self.conv_layer(x)

        # Flatten
        features = features.view(features.size(0), -1)

        # value and advantage
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # estimate Q values
        q_vals = values + (advantages - advantages.mean())

        return q_vals
