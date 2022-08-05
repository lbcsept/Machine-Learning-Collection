
from torch import block_diag, nn  
import torch
import numpy as np

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
    [(1, 0, 1, 256), (3, 1, 1, 512), 4],
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
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(leakyrelu_neg_slope)
        
    def forward(self, x):
        
        return self.activation(self.bn(self.conv(x)))
    
    # def get_out_shape(self, input_shape=(3, 448, 448)):

    #     input = torch.randn(input_shape)
    #     return self.forward(*input).shape


class YoloV1(nn.Module):
    
    def __init__(self, arch_id="24", nclass=20, nbox=2, s_grid=7, image_shape=(448, 448, 3),
            hid_ly=496, head_dropout_p=0.0, lkrelu_slope = 0.1, ncol_coords=5):
        """_summary_
        Args:
            arch_id (str, optional): name of the yolo v1 architecture (only "24" is currently supported). Defaults to "24".
            nclass (int, optional): number of class to predict Defaults to 20.
            nbox (int, optional): number of anchor boxes per grid cell. Defaults to 2.
            s_grid (int, optional): number of grid cell (one value, will be same for height and width). Defaults to 7.
            in_channels (int, optional): number of input channel (3 for RGB images). Defaults to 3.
            image_shape (int, optional): input image shape. Defaults to 3.
            hid_ly (int, optional): hidden layer dimension of the head (in paper 4096). Defaults to 496.
            head_dropout_p (int, optional): dropout proba of the dense layer. Defaults to 0.0.
            lkrelu_slope (int, optional): hidden layer leaky rely slope. Defaults to 0.1.
            ncol_coords (int, optional):  number of columns for 1 box coordinates : 5 = (objectness, x0, y0, h, w). Defaults to 5.
        """
        
        super(YoloV1, self).__init__()
        self.image_shape = image_shape
        self.arch_id=arch_id
        self.architecture = architectures[self.arch_id]
        self.in_channels = self.image_shape[-1]
        self.maxp = nn.MaxPool2d(2)
        self.C = nclass
        self.B = nbox
        self.S = s_grid

        # model hyper params
        self.hid_ly = hid_ly 
        self.head_dropout_p = head_dropout_p
        self.lkrelu_slope = lkrelu_slope
        self.ncol_coords=ncol_coords 

        self.darknet = self._create_backbone()
        self.head =  self._create_head()

    def print_params(self):
        #crepr = super(YoloV1, self).__repr__()
        str_rep = [] #[crepr]
        pparams = ["image_shape", "arch_id", "C", "B", "S", "hid_ly", "head_dropout_p","lkrelu_slope","ncol_coords"]
        params_dict = {} #{p:{getattr(self, p)} for p in pparams }
        for p in pparams:
            params_dict[p]= getattr(self, p)
        print("<>" * 10 + " Models params" + "<>" * 10)
        print(params_dict)
        print("<>" * 10 + " \Models params" + "<>" * 10)

        #return "\n".join(str_rep)

    def _compute_backbone_shape(self):
        """Provide backbone output shape based on model input shape"""
        img_shape = list(self.image_shape)
        input_shape = [1, img_shape[-1] ] + img_shape[:-1]
        return self.darknet(torch.randn(input_shape)).shape

    def _create_backbone(self):

        layers = []
        in_channels = self.in_channels 
        for ly in self.architecture:
            
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
                reps = ly[-1]
                for itt in range(reps):
                    for lly in ly[:-1]:
                        kernel_size, padding, stride,  out_channels = lly
                        layers += [ConvBlock(in_channels=in_channels, out_channels=out_channels, 
                                     kernel_size=kernel_size, stride=stride, padding=padding)]
                        in_channels = out_channels
                    
        model = nn.Sequential(*layers)
        return model

    def _create_head(self,):
        
        # 
        # dims
        last_backbone_ly = self.architecture[-1]
        dim_input = np.product(self._compute_backbone_shape()[1:])
        print(f"head dim input : {dim_input}")
        dim_input = np.product(self._compute_backbone_shape()[1:])

        dim_output = self.S * self.S *  (self.C + self.B * self.ncol_coords)
        
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim_input, self.hid_ly),
            nn.Dropout(self.head_dropout_p),
            nn.LeakyReLU(self.lkrelu_slope),
            nn.Linear(self.hid_ly, dim_output)
        )
        

    def forward(self, x):
        
        backbone = self.darknet(x)
        #print(f"backbone output shape {backbone.shape} ==> {np.prod(backbone[1:].shape)}")
        head = self.head(backbone)
        return backbone, head



if __name__ == "__main__":
    
    def input_shape_from_image_shape(image_shape=(448, 448, 3), batch_size=2):
        image_shape = list(image_shape)
        input_shape = [batch_size] + [image_shape[-1]] + image_shape[:-1]
        return tuple(input_shape)

    print("*"*80 + "\nSTART\n" + "*"*80)
    device = "cpu"

    image_shape = [448, 448, 3]
    
    #print("-" * 5 + f" testing with input shape {image_shape}")
    model = YoloV1(image_shape=image_shape).to(device)
    model.print_params()
    model.eval()
    x = torch.randn(input_shape_from_image_shape(image_shape)).to(device)
    print(f"input shape : {x.shape}")
    backbone, pred = model(x)
    print(f"output shape : {pred.shape}")
    

    image_shape = [448*2, 448, 3]
    model = YoloV1(image_shape=image_shape).to(device)
    model.print_params()
    model.eval()
    x = torch.randn(input_shape_from_image_shape(image_shape)).to(device)
    print(f"input shape : {x.shape}")
    backbone, pred = model(x)
    print(f"output shape : {pred.shape}")


    image_shape = [1024, 448*2, 3]
    model = YoloV1( nclass=5, nbox=2, s_grid=25, image_shape=image_shape).to(device)
    model.print_params()
    model.eval()
    x = torch.randn(input_shape_from_image_shape(image_shape)).to(device)
    print(f"input shape : {x.shape}")
    backbone, pred = model(x)
    print(f"output shape : {pred.shape}")


    image_shape = [1024, 448*2, 3]
    model = YoloV1( nclass=5, nbox=3, s_grid=25, image_shape=image_shape).to(device)
    model.print_params()
    model.eval()
    x = torch.randn(input_shape_from_image_shape(image_shape)).to(device)
    print(f"input shape : {x.shape}")
    backbone, pred = model(x)
    print(f"output shape : {pred.shape}")


    print("*"*80 + "\nEND\n" + "*"*80)