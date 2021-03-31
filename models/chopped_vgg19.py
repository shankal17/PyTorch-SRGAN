import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ChoppedVGG19(nn.Module):
    
    def __init__(self, layer):
        super(ChoppedVGG19, self).__init__()
        self.layers = nn.Sequential(*list(torchvision.models.vgg19(pretrained=True).features.children())[:layer]).eval()
        # We don't want to train this, it's only used to extract features to calculate loss
        for _, param in self.layers.named_parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.layers(x)

    def vgg_loss(self, x, target):
        return F.mse_loss(self.layers(x), self.layers(target))

if __name__ == '__main__':
    cm = ChoppedVGG19(36)
    print(cm.layers)