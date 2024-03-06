import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepEmotion(nn.Module):
    def __init__(self):
        super(DeepEmotion, self).__init__()
        self.feat_extraction = nn.Sequential(
            nn.Conv2d(1,10,3),
            nn.ReLU(True),
            nn.Conv2d(10,10,3),
            nn.MaxPool2d(2,2),
            nn.ReLU(True),
            nn.Conv2d(10,10,3),
            nn.ReLU(True),
            nn.Conv2d(10,10,3),
            nn.BatchNorm2d(10),
            nn.MaxPool2d(2,2),
            nn.ReLU(True),
            nn.Dropout2d(),
        )

        self.fc_de = nn.Sequential(
            nn.Linear(810, 50),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(50, 7)
        )

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(1000, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # initialize the weights/bias with identity transformation               
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))


    def forward(self, input):
        feat = self.feat_extraction(input)
        xs = self.localization(input)
        xs = xs.view(-1, 1000)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, feat.size())
        sample = F.grid_sample(feat, grid)
        sample = sample.view(-1, 810)
        out = self.fc_de(sample)
        return out









