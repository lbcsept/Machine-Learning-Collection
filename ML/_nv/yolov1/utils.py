


architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]



def cnn__out_dims(architecture, input_dim=(448, 448, 3)):
    in_dim = input_dim

    for ly in architecture:
        
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
            reps = ly.pop(-1)
            for itt in range(reps):
                for lly in ly:
                    kernel_size, padding, stride,  out_channels = lly
                    layers += [ConvBlock(in_channels=in_channels, out_channels=out_channels, 
                                    kernel_size=kernel_size, stride=stride, padding=padding)]
                    in_channels = out_channels        if 

