
import torch.nn as nn
import torch.nn.functional as F

class BasicCNN(nn.Module):
    
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        # input 1x28x28
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3,stride=1,padding=1) #same ==> 8x28x28
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2) # => 14 X 14
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3,stride=1,padding=1) #same ==> 16x14x14
        #self.pool = nn.MaxPool2d(kernel_size=2,stride=2) # => 14 X 14

        self.fc1 = nn.Linear(16*7*7, 256) # 16*7*7
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        #print(x.shape)
        x = F.relu(self.pool(self.conv1(x)))
        #print(x.shape)
        x = F.relu(self.pool(self.conv2(x)))
        #print(x.shape)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


