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

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def test(pic,label,epsilon = 0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    label = torch.tensor(label)
    label = label.unsqueeze(0)
    label = label.to(device)
    pic = pic.to(device)
    pic.requires_grad = True
    dist = torch.load("./fashion_mnist.pth")
    model.load_state_dict(dist)
    model.eval()
    output = model(pic)
    init_pred = output.argmax(dim=1)[0]
    loss = F.nll_loss(output,label)
    model.zero_grad()
    loss.backward()
    data_grad = pic.grad.data
    perturbed_data = fgsm_attack(pic, epsilon, data_grad)
    attack = model(perturbed_data)
    final_pred = attack.max(1, keepdim=True)[1]
    print("output:{}  attack:{}".format(text_labels[init_pred],text_labels[final_pred]))
    result =  label != final_pred
    test_result =  label == init_pred

    result = result.to("cpu").numpy()
    print("test success: {}  attack success:{}".format(test_result[0],result[0][0]))

    pic = pic.to("cpu").detach().numpy()
    perturbed_data = perturbed_data.to("cpu").detach().numpy()
    plt.subplot(1,2,1)
    plt.imshow(pic[0,0, ...])

    plt.subplot(1, 2, 2)
    perturbed_data = perturbed_data.squeeze(0)
    plt.imshow(perturbed_data[0,...])
    plt.show()


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
    test(picture,my_dataset[idx[0]][1],epsilon = 0.1)


if __name__ == '__main__':
    for i in range(10):
        main()
