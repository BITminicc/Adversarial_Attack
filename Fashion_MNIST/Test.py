import random
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
# 定义超参数
num_epochs = 10
text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def test(pic,label):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    pic = pic.to(device)
    dist = torch.load("./fashion_mnist.pth")
    model.load_state_dict(dist)
    model.eval()
    with torch.no_grad():
        output = model(pic)
        output = output.argmax(dim=1)[0]
        print("output:{}".format(text_labels[output]))
        result = label==output
        result = result.to("cpu").numpy()
        print("result: ",result)

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])
    my_dataset = datasets.FashionMNIST('./FashionMnist', train=True, download=True,
                                     transform=transform)

    idx = random.sample(range(0, len(my_dataset)), 1)

    pic = my_dataset[idx[0]][0].numpy()
    print('random int:{}   label is:{} '.format(idx[0], text_labels[my_dataset[idx[0]][1]]))
    pic = torch.tensor(pic)

    picture = torch.unsqueeze(pic,dim=0)
    test(picture,my_dataset[idx[0]][1])
    plt.imshow(pic[0, ...])
    plt.show()


if __name__ == '__main__':
    for i in range(10):
        main()
