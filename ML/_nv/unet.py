
from grpc import insecure_channel
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    
    def __init__(self, 
                in_channels: int,
                out_channels: int,
                kernel_size=3,
                stride=1,
                padding=0,
                dilation=1,
                bias: bool = False,
                padding_mode: str = 'zeros', 
                batch_norm=True, 
                activation=F.relu):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                            stride=stride, padding=padding, dilation=dilation, bias=bias, padding_mode=padding_mode)
        self.activation = activation
        self.batch_norm = batch_norm
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        
        x = self.conv(x)
        x = self.bn(x) if self.batch_norm else x
        x = self.activation(x) if self.activation is not None else x
        
        return x


class UnetBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels=None, widthf = 2, kernel_size=3, padding = 0, depth= 2):
        super().__init__()
        self.in_channels = in_channels
        self.depth = depth
        out_channels = widthf*in_channels if out_channels is None else out_channels
        #self.convs = nn.Sequential(*[Conv(in_channels=in_channels, out_channels = out_channels, kernel_size=kernel_size,
        #                  padding = padding) for _ in range(depth)])
        convl = [Conv(in_channels=in_channels, out_channels = out_channels, kernel_size=kernel_size,padding = padding)]
        convl += [Conv(in_channels=out_channels, out_channels = out_channels, kernel_size=kernel_size,padding = padding) 
                for _ in range(depth - 1)] 
        self.convs = nn.Sequential(*convl)
    
    def forward(self, x):
        
        return self.convs(x)

class ExpansionBlock(nn.Module):
    def __init__(self, in_channels,out_channels=None, widthf = 2):
        super().__init__()
        
        out_channels = int(in_channels/widthf) if out_channels is None else out_channels 
        self.convtr = nn.ConvTranspose2d(in_channels=in_channels,  out_channels=out_channels, )
        
    def forward(self):
        pass
        
class Unet(nn.Module):
    
    def __init__(self, in_channels=1, class_num=5):
        super().__init__()
        in_c = [in_channels, 64, 128, 256,  512, 1024]

        self.cb1 = UnetBlock(in_channels=in_c[0], out_channels=in_c[1])
        self.cb2 = UnetBlock(in_channels=in_c[1], out_channels=in_c[2])
        self.cb3 = UnetBlock(in_channels=in_c[2], out_channels=in_c[3])
        self.cb4 = UnetBlock(in_channels=in_c[3], out_channels=in_c[4])
        self.cb5 = UnetBlock(in_channels=in_c[4], out_channels=in_c[5])
        self.convt4 = nn.ConvTranspose2d(in_channels=in_c[5],  out_channels=in_c[4], kernel_size=2, stride=2, padding=0)
        self.convt3 = nn.ConvTranspose2d(in_channels=in_c[4],  out_channels=in_c[3], kernel_size=2, stride=2, padding=0)
        self.convt2 = nn.ConvTranspose2d(in_channels=in_c[3],  out_channels=in_c[2], kernel_size=2, stride=2, padding=0)
        self.convt1 = nn.ConvTranspose2d(in_channels=in_c[2],  out_channels=in_c[1], kernel_size=2, stride=2, padding=0)
        
        self.exp4 = UnetBlock(in_channels=in_c[5], out_channels=in_c[4])
        self.exp3 = UnetBlock(in_channels=in_c[4], out_channels=in_c[3])
        self.exp2 = UnetBlock(in_channels=in_c[3], out_channels=in_c[2])
        self.exp1 = UnetBlock(in_channels=in_c[2], out_channels=in_c[1])
        self.expcl = UnetBlock(in_channels=in_c[1], out_channels=class_num, padding=1, depth=1)

        self.maxpool= nn.MaxPool2d(2,2)
    
    def _crop_concatenate(self, X, x):
        h = x.shape[-2]
        delta = X.shape[-2] - h
        X = X[...,int(delta/2):int(delta/2)+h, int(delta/2):int(delta/2)+h]
        return torch.cat([X, x], dim=1)
        
    def forward(self, x):
        print(x.shape)
        x = self.cb1(x)
        sk_x1 = x
        x = self.maxpool(x)
        print(x.shape)
        x = self.cb2(x)
        sk_x2 = x
        x = self.maxpool(x)
        print(x.shape)
        x = self.cb3(x)
        sk_x3 = x
        x = self.maxpool(x)
        print(x.shape)
        x = self.cb4(x)
        sk_x4 = x
        x = self.maxpool(x)
        print(x.shape)
        x = self.cb5(x)
        print(x.shape)
        x = self.convt4(x)
        print(f"after convt4 {x.shape}")
        x = self._crop_concatenate(sk_x4, x)
        print(f"after concatenate 4 {x.shape}")

        x = self.exp4(x)
        x = self.convt3(x)
        x = self._crop_concatenate(sk_x3, x)
        print(f"after concatenate 3 {x.shape}")

        x = self.exp3(x)
        x = self.convt2(x)
        x = self._crop_concatenate(sk_x2, x)
        print(f"after concatenate 2 {x.shape}")

        x = self.exp2(x)
        x = self.convt1(x)
        x = self._crop_concatenate(sk_x1, x)
        print(f"after concatenate 1 {x.shape}")

        x = self.exp1(x)
        print(f"before classes {x.shape}")


        #print(sk_x1.shape)
        return self.expcl(x)
    
        
if __name__ == "__main__":
    
    # torch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    imgt = torch.randn((2, 3, 572, 572)).to(device)
    print(imgt.shape)
    model = Unet(in_channels=3).to(device)
    print(model)
    x = model(imgt)
    print(f"x.shape {x.shape}")