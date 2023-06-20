import torch
import torch.nn as nn
from torchvision.models import resnet50


class SiameseNetwork(nn.Module):
    """ Loads the pretrained Resnet18 model. Set mode for fine-tuning """

    def __init__(self, lastLayer=False, pretrained=True):
        super(SiameseNetwork, self).__init__()

        self.lastLayer = lastLayer
        self.net_parameters = []  # List of parameters requiring grad

        #self.model_conv = torchvision.models.resnet18(pretrained=pretrained)
        self.model_conv = resnet50(pretrained=pretrained)
        #self.model_conv = torchvision.models.resnet101(pretrained=pretrained)

        self.fc = nn.Sequential(
            nn.Linear(2000, 1),
            nn.Sigmoid()
        )

        if pretrained:
            # freeze all parameters in the model
            for param in self.model_conv.parameters():
                param.requires_grad = False

            for param in self.model_conv.fc.parameters():
                param.requires_grad = True
                self.net_parameters.append(param)

        if self.lastLayer:
            self.out_last = self.model_conv.fc.out_features

            self.extraL = nn.Linear(self.out_last, 1)
            for param in self.extraL.parameters():
                param.requires_grad = True
                self.net_parameters.append(param)

    def forward_once(self, x):
        output = self.model_conv(x)
        # if self.lastLayer:
        #     output = self.extraL(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        output = torch.cat((output1, output2), 1)
        output = self.fc(output)

        return output