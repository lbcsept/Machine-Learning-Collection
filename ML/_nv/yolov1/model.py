
from torch import block_diag, nn  
import torch


# kernel size, padding, stride,  out_channels
architectures = {"24":
    [
    (7, 1, 2, 64),
    "M",
    (3, 1, 1, 192),
    "M",
    (1, 0, 1, 128),
    (3, 1, 1, 256),
    (1, 0, 1, 256),
    (3, 1, 1, 512),
    "M",
    [(1, 0, 1, 256), (3, 1, 1, 512), 4]
    (1, 0, 1, 512),
    (3, 1, 1, 1024),
    "M",
    [(1, 0, 1, 512), (3, 1, 1, 1024), 2],
    (3, 1, 1, 1024),
    (3, 1, 2, 1024),
    (3, 1, 1, 1024),
    (3, 1, 1, 1024)   
    ]
}

class ConvBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, bias= False, leakyrelu_neg_slope=0.1, **kwargs):
        
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=bias, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels=out_channels)
        self.activation = nn.LeakyReLU(leakyrelu_neg_slope)
        
    def forward(self, x):
        
        return self.activation(self.bn(self.conv(x)))
        


class YoloV1(nn.Module):
    
    def __init__(self, arch_id="24", nclass=20, nbox=2, s_grid=7, in_channels=3):
        """_summary_

        Args:
            arch_id (str, optional): name of the yolo v1 architecture (only "24" is currently supported). Defaults to "24".
            nclass (int, optional): number of class to predict Defaults to 20.
            nbox (int, optional): number of anchor boxes per grid cell. Defaults to 2.
            s_grid (int, optional): number of grid cell (one value, will be same for height and width). Defaults to 7.
            in_channels (int, optional): _description_. Defaults to 3.
        """
        
        super(YoloV1, self).__init__()
        self.arch_id=arch_id
        self.in_channels = 3
        self.maxp = nn.MaxPool2d(2)
        self.darknet = self._create_backbone(self.in_channels)
        self.C = nclass
        self.B = nbox
        self.S = s_grid
        
    
    def _create_backbone(self):
        layers = []
        in_channels = self.in_channels 
        for ly in architectures[self.arch_id]:
            
            if isinstance(ly, tuple):
                # cnn blocks
                kernel_size, padding, stride,  out_channels = ly
                layers += [ConvBlock(in_channels=in_channels, out_channels=out_channels, 
                                     kernel_size=kernel_size, stride=stride, padding=padding)]
                
                in_channels = out_channels
            elif isinstance(ly, str):
                # max pooling 2
                layers += [self.maxp]
                
            elif isinstance(ly, list):
                reps = ly[-1].pop()
                for itt in range(reps):
                    for lly in range(len(ly)):
                        kernel_size, padding, stride,  out_channels = ly
                        layers += [ConvBlock(in_channels=in_channels, out_channels=out_channels, 
                                     kernel_size=kernel_size, stride=stride, padding=padding)]
                        in_channels = out_channels
                    
        model = nn.Sequential(*layers)
        
    def _create_fcs(self, x, ncol_coords=5, hidden_ly = 1024):
        
        # 
        # dims 
        dim_per_sample = self.S * self.S *  (self.C + self.B * ncol_coords)
        
        nn.Sequential([nn.Flatten(),
                       nn.Linear()])
        
        return x
        

    def forward(self, x):
        
        x = self.darknet(x)
        
        return self.create_fcs(x)


if __name__ == "__main__":
    device = "cpu"
    model = YoloV1().to(device)
    model.eval()

    x = torch.randn(2, 3, 448, 448).to(device)
    pred = model(x)
    print(pred.shape)