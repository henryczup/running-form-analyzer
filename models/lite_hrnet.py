

import cv2
import numpy as np
import torch
from torch import nn

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=2):
        super(ShuffleUnit, self).__init__()
        self.groups = groups
        self.conv = ConvBNReLU(in_channels, out_channels, kernel_size=1, groups=groups)

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.conv(x)
        x = x.view(b, self.groups, c // self.groups, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(b, -1, h, w)
        return x

class LiteHRModule(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(LiteHRModule, self).__init__()
        self.branch1 = nn.Sequential(
            ConvBNReLU(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels),
            ConvBNReLU(in_channels, out_channels, kernel_size=1)
        )
        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, kernel_size=1),
            ConvBNReLU(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=out_channels),
            ConvBNReLU(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.branch1(x) + self.branch2(x)

class LiteHRNet(nn.Module):
    def __init__(self, num_joints):
        super(LiteHRNet, self).__init__()
        self.conv1 = ConvBNReLU(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBNReLU(32, 64, kernel_size=3, stride=2, padding=1)

        self.stage1 = nn.Sequential(
            LiteHRModule(64, 64),
            LiteHRModule(64, 64),
            LiteHRModule(64, 64),
            LiteHRModule(64, 64)
        )

        self.stage2 = nn.Sequential(
            LiteHRModule(64, 128, stride=2),
            LiteHRModule(128, 128),
            LiteHRModule(128, 128),
            LiteHRModule(128, 128)
        )

        self.stage3 = nn.Sequential(
            LiteHRModule(128, 256, stride=2),
            LiteHRModule(256, 256),
            LiteHRModule(256, 256),
            LiteHRModule(256, 256)
        )

        self.final_layer = nn.Conv2d(256, num_joints, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.final_layer(x)
        return x

class LiteHRNetModel:
    def __init__(self, config):
        self.model = LiteHRNet(num_joints=17)  # Assuming 17 joints for COCO format
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image):
        image_tensor = self._preprocess(image)
        with torch.no_grad():
            output = self.model(image_tensor)
        keypoints = self._postprocess(output)
        return keypoints

    def _preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))  # Resize to a fixed size
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        image_tensor = image_tensor / 255.0  # Normalize to [0, 1]
        return image_tensor

    def _postprocess(self, output):
        heatmaps = output.squeeze().cpu().numpy()
        keypoints = []
        for i in range(heatmaps.shape[0]):
            heatmap = heatmaps[i]
            y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            confidence = heatmap[y, x]
            keypoints.append([x, y, confidence])
        return np.array(keypoints).reshape(1, -1, 3)