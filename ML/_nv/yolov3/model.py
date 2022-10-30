from turtle import forward
from torch import nn



class convBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bn=True, **kwargs):
        
        super().__init__()
     
        self.conv = nn.Conv2d(in_channels=in_channels, 
                  out_channels=out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding,
                  bias= not bn, **kwargs)
        
        self.bn = nn.BatchNorm2d(num_features=out_channels) if bn else None
        self.leaky = nn.LeakyReLU(0.1)
        
    def forward(self, X):
        
        if self.bn is not None:
            return self.leaky(self.bn(self.conv(X)))

class YoloV3(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        
    
    def forward(self):
        pass
    