import random
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
# 定义超参数
num_epochs = 10

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def test(pic):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    pic = pic.to(device)
    dist = torch.load("./mnist.pth")
    model.load_state_dict(dist)
    model.eval()
    with torch.no_grad():
        output = model(pic)
        output = output.argmax(dim=1)[0]
        print("output:{}".format(output))

def main():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    my_dataset = datasets.MNIST('./data', train=False,
                                transform=transform)

    idx = random.sample(range(0, len(my_dataset)), 1)

    pic = my_dataset[idx[0]][0].numpy()
    print('random int:{}   label is:{} '.format(idx[0], my_dataset[idx[0]][1]))
    pic = torch.tensor(pic)
    picture = torch.unsqueeze(pic,dim=0)
    test(picture)
    plt.imshow(pic[0, ...])
    plt.show()


if __name__ == '__main__':
    for i in range(10):
        main()
