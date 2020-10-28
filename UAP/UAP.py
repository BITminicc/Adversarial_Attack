import model
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torch import optim
import torchvision
import torchvision.transforms as transforms
import os
from torch.autograd import Variable
import copy
from torch.autograd.gradcheck import zero_gradients
import numpy as np
import random

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

class Trainer:
    def __init__(self,device):
        self.device = device
        self.net = model.ConvNet()
        self.net.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.01)
        self.n_epochs = 10

    def train(self,trainloader,testloader):
        accuracy = 0
        self.net.train()
        for epoch in range(self.n_epochs):
            running_loss = 0.0
            print_every = 200  # mini-batches
            for i, (inputs, labels) in enumerate(trainloader, 0):
                # Transfer to GPU
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                # print statistics
                running_loss += loss.item()
                if (i % print_every) == (print_every-1):
                    print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/print_every))
                    running_loss = 0.0
            # Print accuracy after every epoch
            accuracy = compute_accuracy(self.net, testloader,self.device)
            print('Accuracy of the network on the 10000 test images: %d %%' % (100 * accuracy))
        print('Finished Training')
        torch.save(self.net.state_dict(),"./net.pth")
        return accuracy

def compute_accuracy(net, testloader,device):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def project_perturbation(data_point,p,perturbation  ):

    if p == 2:
        perturbation = perturbation * min(1, data_point / np.linalg.norm(perturbation.flatten(1)))
    elif p == np.inf:
        perturbation = np.sign(perturbation) * np.minimum(abs(perturbation), data_point)
    return perturbation

def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),  # Transform to tensor
        transforms.Normalize((0.5,), (0.5,))  # Min-max scaling to [-1, 1]
    ])

    data_dir = os.path.join("./", 'fashion_mnist')
    print('Data stored in %s' % data_dir)
    trainset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False)
    return trainloader,testloader

def generate(accuracy ,trainset, testset, net, delta=0.2, max_iter_uni=np.inf, xi=10, p=np.inf, num_classes=10, overshoot=0.2, max_iter_df=20):
    '''
    :param trainset: Pytorch Dataloader with train data
    :param testset: Pytorch Dataloader with test data
    :param net: Network to be fooled by the adversarial examples
    :param delta: 1-delta represents the fooling_rate, and the objective
    :param max_iter_uni: Maximum number of iterations of the main algorithm
    :param p: Only p==2 or p==infinity are supported
    :param num_class: Number of classes on the dataset
    :param overshoot: Parameter to the Deep_fool algorithm
    :param max_iter_df: Maximum iterations of the deep fool algorithm
    :return: perturbation found (not always the same on every run of the algorithm)
    '''

    net.eval()
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # Importing images and creating an array with them
    img_trn = []
    for image in trainset:
        for image2 in image[0]:
            img_trn.append(image2.numpy())

    img_tst = []
    for image in testset:
        for image2 in image[0]:
            img_tst.append(image2.numpy())

    # Setting the number of images to 300 ? 100
    # (A much lower number than the total number of instances on the training set)
    # To verify the generalization power of the approach
    num_img_trn = 100
    index_order = np.arange(num_img_trn)

    # Initializing the perturbation to 0s
    v=np.zeros([28,28])
    #Initializing fooling rate and iteration count
    fooling_rate = 0.0
    iter = 0
    # Transformers to be applied to images in order to feed them to the network
    transformer = transforms.ToTensor()
    transformer1 = transforms.Compose([
        transforms.ToTensor(),
    ])
    transformer2 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(28),
    ])
    fooling_rates=[0]
    accuracies = []
    accuracies.append(accuracy)
    total_iterations = [0]
    # Begin of the main loop on Universal Adversarial Perturbations algorithm
    while fooling_rate < 1-delta and iter < max_iter_uni:
        np.random.shuffle(index_order)
        print("Iteration  ", iter)

        for index in index_order:
            v = v.reshape((v.shape[0], -1))

            # Generating the original image from data
            cur_img = Image.fromarray(img_trn[index][0])
            cur_img1 = transformer1(transformer2(cur_img))[np.newaxis, :].to(device)

            # Feeding the original image to the network and storing the label returned
            r2 = (net(cur_img1).max(1)[1]).to(device)
            torch.cuda.empty_cache()

            # Generating a perturbed image from the current perturbation v and the original image
            per_img = Image.fromarray(transformer2(cur_img)+v.astype(np.uint8))
            per_img1 = transformer1(transformer2(per_img))[np.newaxis, :].to(device)

            # Feeding the perturbed image to the network and storing the label returned
            r1 = (net(per_img1).max(1)[1]).to(device)
            torch.cuda.empty_cache()

            # If the label of both images is the same, the perturbation v needs to be updated
            if r1 == r2:
                print(">> k =", np.where(index==index_order)[0][0], ', pass #', iter, end='      ')

                # Finding a new minimal perturbation with deepfool to fool the network on this image
                dr, iter_k, label, k_i, pert_image = deepfool(per_img1[0], net, num_classes=num_classes, overshoot=overshoot, max_iter=max_iter_df)

                # Adding the new perturbation found and projecting the perturbation v and data point xi on p.
                if iter_k < max_iter_df-1:

                    v[:, :] += dr[0,0, :, :]

                    v = project_perturbation( xi, p,v)

        iter = iter + 1

        # Reshaping v to the desired shape
        v = v.reshape((v.shape[0], -1, 1))

        with torch.no_grad():

            # Compute fooling_rate
            labels_original_images = torch.tensor(np.zeros(0, dtype=np.int64))
            labels_pertubed_images = torch.tensor(np.zeros(0, dtype=np.int64))

            i = 0
            # Finding labels for original images
            for batch_index, (inputs, _) in enumerate(testset):
                i += inputs.shape[0]
                inputs = inputs.to(device)
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                labels_original_images = torch.cat((labels_original_images, predicted.cpu()))
            torch.cuda.empty_cache()
            correct = 0
            # Finding labels for perturbed images
            for batch_index, (inputs, labels) in enumerate(testset):
                #inputs = inputs.to(device)
                inputs += transformer(v).float()
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                labels_pertubed_images = torch.cat((labels_pertubed_images, predicted.cpu()))
                correct += (predicted == labels).sum().item()
            torch.cuda.empty_cache()


            # Calculating the fooling rate by dividing the number of fooled images by the total number of images
            fooling_rate = float(torch.sum(labels_original_images != labels_pertubed_images))/float(i)

            print()
            print("FOOLING RATE: ", fooling_rate)
            fooling_rates.append(fooling_rate)
            accuracies.append(correct / i)
            total_iterations.append(iter)
    return v,fooling_rates,accuracies,total_iterations

def deepfool(image, net, num_classes, overshoot, max_iter):

    """
       :param image:
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """

    #is_cuda = torch.cuda.is_available()
    is_cuda = False
    if is_cuda:
        #pass
        image = image.cuda()
        net = net.cuda()

    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = f_image.argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
       # print(image.shape)
       # print(x.view(1,1,image.shape[0],-1).shape)
        fs = net.forward(x.view(1,1,image.shape[1],-1))
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    return (1+overshoot)*r_tot, loop_i, label, k_i, pert_image


def pretrain():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(device=device)
    trainset, testset = load_data()
    accuracy = trainer.train(trainset, testset)
    return accuracy

def load():
    device = torch.device("cpu")
    dist = torch.load("./net.pth")
    trainer = Trainer(device=device)
    trainer.net.load_state_dict(dist)
    trainset, testset = load_data()
    accuracy = compute_accuracy(trainer.net,testset,device)
    return  accuracy,trainset, testset,trainer

def attack():
    accuracy, trainset, testset, trainer = load()

    v, fooling_rates, accuracies, total_iterations = \
        generate(accuracy, trainset, testset, trainer.net)

    plt.title("Fooling Rates over Universal Iterations")
    plt.xlabel("Universal Algorithm Iter")
    plt.ylabel("Fooling Rate on test data")
    plt.plot(total_iterations, fooling_rates)
    plt.show()

    plt.title("Accuracy over Universal Iterations")
    plt.xlabel("Universal Algorithm Iter")
    plt.ylabel("Accuracy on Test data")
    plt.plot(total_iterations, accuracies)
    plt.show()

    v = v.squeeze()
    np.save("filename.npy", v)

def test():
    temp = np.load("filename.npy", encoding='bytes', allow_pickle=True)
    device = torch.device("cpu")
    dist = torch.load("./net.pth")
    trainer = Trainer(device=device)
    trainer.net.load_state_dict(dist)

    transform = transforms.Compose([
        transforms.ToTensor(),  # Transform to tensor
        transforms.Normalize((0.5,), (0.5,))  # Min-max scaling to [-1, 1]
    ])
    transformer1 = transforms.Compose([
        transforms.ToTensor(),
    ])
    transformer2 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(28),
    ])
    data_dir = os.path.join("./", 'fashion_mnist')

    testset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    trainer.net.eval()
    idx = random.sample(range(0, len(testset)), 1)
    with torch.no_grad():

        images = testset[idx[0]][0]
        temp = torch.tensor(temp)
        temp = temp.unsqueeze(0)
        image2 = images + temp
        labels = testset[idx[0]][1]
        plt.subplot(1, 3, 1)
        plt.imshow(images[0,...],cmap='gray')

        images = images.unsqueeze(0)
        outputs = trainer.net(images)
        _, predicted = torch.max(outputs.data, 1)
        print("init_out: {} label:{}".format(predicted.data[0],labels))

        plt.subplot(1, 3, 2)
        plt.imshow(image2[0, ...],cmap='gray')
        image2 = image2.unsqueeze(0)
        image2 = torch.tensor(image2, dtype=torch.float32)
        outputs = trainer.net(image2)
        _, predicted = torch.max(outputs.data, 1)
        print("attack_out: {} label:{}".format(predicted.data[0], labels))

    #print(temp)
    # plt.imshow(v)
    plt.subplot(1, 3, 3)
    plt.imshow(temp[0,...],cmap='gray')

    plt.show()

def main():

    #pretrain()

    #attack()

    test()




if __name__ == "__main__":
    main()
