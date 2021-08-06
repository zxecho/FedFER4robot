"""VGG11/13/16/19 in Pytorch."""
import torch.nn as nn
import torch.nn.functional as F

cfg = {
    'VGGs': [32, 'M', 64, 'M', 64, 'M', 128, 'M'],
    'VGGs3': [32, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M'],
    'VGGs2': [32, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M'],
    'VGGs1': [32, 'M', 64, 'M', 128, 128, 'M', 128, 128, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, in_chanels, vgg_name, class_num=7):
        super(VGG, self).__init__()
        self.in_chanels = in_chanels
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, class_num)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_chanels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
