import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
import numpy as np

# simple
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(64 * 8 * 8, 128)
#         self.fc2 = nn.Linear(128, 100)
#
#     def forward(self, x):
#         x = self.pool(nn.functional.relu(self.conv1(x)))
#         x = self.pool(nn.functional.relu(self.conv2(x)))
#         x = x.view(-1, 64 * 8 * 8)
#         x = nn.functional.relu(self.fc1(x))
#         x = nn.functional.softmax(self.fc2(x), dim=1)
#         return x

# # Define the CNN model self-defined
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.layers1 = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),# input 32*32*3 output 32*32*32
#             nn.BatchNorm2d(32),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2, stride=2),#output 16*16*32
#         )
#         self.layers2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),# input 16*16*32 output 16*16*64
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#         )
#         self.layers3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),# input 16*16*64 output 16*16*128
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             nn.Conv2d(128, 256, kernel_size=3, padding=1), # input 16*16*128 output 16*16*256
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#             nn.Conv2d(256, 512, kernel_size=3, padding=1),# input 16*16*256 output 16*16*512
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#             nn.Dropout(0.2),
#         )
#         self.layers4 = nn.Sequential(
#             nn.Linear(512 * 16 * 16, 512),
#             nn.ReLU(True),
#             nn.Dropout(0.2),
#             nn.Linear(512, 100),
#         )
#
#
#     def forward(self, x):
#         x = self.layers1(x)
#         x = self.layers2(x)
#         x = self.layers3(x)
#         x = x.view(-1, 512 * 16 * 16)
#         x = self.layers4(x)
#         return x

# #优化1
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=5,padding=2),#3代表输入通道，16代表输出通道（16个卷积核）
#             nn.BatchNorm2d(16),#参数数值为输入维度
#             nn.ReLU(),
#         )
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(16, 32, kernel_size=5, padding=2),  # 3代表输入通道，16代表输出通道（16个卷积核）
#             nn.BatchNorm2d(32),  # 参数数值为输入维度
#             nn.ReLU(),
#         )
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=5, padding=2),  # 3代表输入通道，16代表输出通道（16个卷积核）
#             nn.BatchNorm2d(64),  # 参数数值为输入维度
#             nn.ReLU(),
#         )
#         self.layer4 =nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 3代表输入通道，16代表输出通道（16个卷积核）
#             nn.BatchNorm2d(64),  # 参数数值为输入维度
#             nn.ReLU()
#         )
#         self.fc1 = nn.Linear(64 * 8 * 8, 100)
#         #池化层
#         self.pool = nn.MaxPool2d(2)
#
#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.pool(self.layer2(out))
#         out = self.pool(self.layer3(out))
#         out = self.layer4(out)
#         out = out.reshape(out.size(0), -1)
#         out = self.fc1(out)
#         return out

#过拟合尝试1
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5,padding=2),#3代表输入通道，16代表输出通道（16个卷积核）
            nn.BatchNorm2d(64),#参数数值为输入维度
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),  # 3代表输入通道，16代表输出通道（16个卷积核）
            nn.BatchNorm2d(128),  # 参数数值为输入维度
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, padding=2),  # 3代表输入通道，16代表输出通道（16个卷积核）
            nn.BatchNorm2d(256),  # 参数数值为输入维度
            nn.ReLU(),
        )
        self.layer4 =nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),  # 3代表输入通道，16代表输出通道（16个卷积核）
            nn.BatchNorm2d(64),  # 参数数值为输入维度
            nn.ReLU()
        )
        self.fc1 = nn.Linear(64 * 8 * 8, 100)
        #池化层
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.pool(self.layer2(out))
        out = self.pool(self.layer3(out))
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        return out

import torch
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def load_data():
    # 定义数据增强
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 加载数据集
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=8,pin_memory=True)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=8,pin_memory=True)

    # 加载验证集
    validset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    validloader = torch.utils.data.DataLoader(validset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    # 定义类别名称
    classes = tuple(trainset.classes)
    return trainloader,testloader,validloader,classes



def get_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = CNN().to(device)
    return net,device

def get_config(net):
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    return criterion,optimizer

def train_model(net, criterion, optimizer, trainloader, valloader, device,epochs=30):
    # Train the model
    net.to(device)
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    train_loss_iter_list = []
    train_acc_iter_list = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        print("Epoch[{}/{}]:".format(epoch + 1, epochs))
        running_train_loss = 0.0
        running_train_acc = 0.0
        running_train_iter_loss = 0.0
        running_train_iter_acc = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            acc = accuracy(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_train_iter_loss += loss.item()
            running_train_iter_acc += acc.item()
            running_train_loss += loss.item()
            running_train_acc += acc.item()

            if i % 100 == 99:  # print every 100 mini-batches
                print('[%d, %5d] train_loss: %.3f, train_accuracy: %.3f' %
                      (epoch + 1, i + 1, running_train_iter_loss / 100, running_train_iter_acc / 100))
                train_loss_iter_list.append(running_train_iter_loss / 100)
                train_acc_iter_list.append(running_train_iter_acc / 100)
                running_train_iter_loss = 0.0
                running_train_iter_acc = 0.0

        train_loss_list.append(running_train_loss / len(trainloader))
        train_acc_list.append(running_train_acc / len(trainloader))
        print('Epoch[ %d / %d ] : train_loss: %.3f, train_accuracy: %.3f' % (epoch + 1, epochs, running_train_loss / len(trainloader), running_train_acc / len(trainloader)))


        # Evaluate the model on the validation set
        running_val_loss = 0.0
        running_val_acc = 0.0
        with torch.no_grad():
            for data in valloader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)
                val_loss = criterion(outputs, labels)
                val_acc = accuracy(outputs, labels)
                running_val_loss += val_loss.item()
                running_val_acc += val_acc.item()
        val_loss_list.append(running_val_loss / len(valloader))
        val_acc_list.append(running_val_acc / len(valloader))
        print('Epoch[ %d / %d ] : val_loss: %.3f, val_accuracy: %.3f' %
              (epoch+1, epochs, running_val_loss / len(valloader), running_val_acc / len(valloader)))

    print('Finished Training')

    # Plot the loss and accuracy curves
    plt.plot(train_loss_iter_list, label='train')
    # plt.plot(val_loss_list, label='val')
    plt.title('Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(train_acc_iter_list, label='train')
    # plt.plot(val_acc_list, label='val')
    plt.title('Accuracy Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(train_loss_list, label='train')
    plt.plot(val_loss_list, label='val')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(train_acc_list, label='train')
    plt.plot(val_acc_list, label='val')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def test_model(net,testloader,criterion,device):
    # Test the model
    correct = 0
    total = 0
    test_loss = 0.0
    predictions=[]
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss += criterion(outputs, labels).item()
            predictions.append(predicted.cpu().numpy())

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    print('Test loss: %.3f' % (test_loss / len(testloader)))

    predictions = np.concatenate(predictions)
    class_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                   'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
                   'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup',
                   'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
                   'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
                   'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter',
                   'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum',
                   'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',
                   'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper',
                   'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle',
                   'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
    # 随机选择 5*5 张图像进行可视化
    images, labels = iter(testloader).__next__()
    outputs = net(images.to(device))
    _, predicted = torch.max(outputs, 1)

    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        # 显示图像
        ax.imshow(np.transpose(images[i].cpu().numpy(), (1, 2, 0)))

        # 判断预测结果是否与真实标签相同
        predicted_label = class_names[predicted[i]]
        true_label = class_names[labels[i]]
        if predicted_label == true_label:
            label_color = 'green'
            label_text = 'Correct'
        else:
            label_color = 'red'
            label_text = 'Wrong'

        # 设置图像标题为分类名称和结果标签
        ax.set_title(f"Predicted: {predicted_label} \n results: {label_text} \n True: {true_label}", color=label_color)

    plt.tight_layout()
    plt.show()




if __name__ == '__main__':
    trainloader, testloader,validloader,classes=load_data()
    net,device = get_model()
    print('current device:  ', device)
    criterion,optimizer=get_config(net)
    train_model(net, criterion, optimizer, trainloader, validloader, device)
    test_model(net, testloader, criterion, device)







# def load_data():
#     # Load the CIFAR100 dataset and define the data loaders
#     train_transform = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#
#     test_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#     trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
#     testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
#     return trainloader,testloader
#
#
# def get_config():
#     # Define the model, loss function, and optimizer
#     net = CNN()
#
#     model_params = list(net.parameters())
#     print("Number of parameters:", len(model_params))
#
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     return net,criterion,optimizer,device
#
# import torch
# def accuracy(outputs, labels):
#     _, preds = torch.max(outputs, dim=1)
#     return torch.tensor(torch.sum(preds == labels).item() / len(preds))
#
#
# import matplotlib.pyplot as plt
#
#
# def train_model(net, criterion, optimizer, trainloader, device):
#     # Train the model
#     net.to(device)
#     loss_list = []
#     acc_list = []
#     for epoch in range(10):  # loop over the dataset multiple times
#         print("Epoch[{}/{}]:".format(epoch + 1, 10))
#         running_loss = 0.0
#         running_acc = 0.0
#         for i, data in enumerate(trainloader, 0):
#             # get the inputs
#             inputs, labels = data[0].to(device), data[1].to(device)
#
#             # zero the parameter gradients
#             optimizer.zero_grad()
#
#             # forward + backward + optimize
#             outputs = net(inputs)
#             loss = criterion(outputs, labels)
#             acc = accuracy(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             # print statistics
#             running_loss += loss.item()
#             running_acc += acc.item()
#             if i % 100 == 99:  # print every 100 mini-batches
#                 print('[%d, %5d] loss: %.3f, accuracy: %.3f' %
#                       (epoch + 1, i + 1, running_loss / 100, running_acc / 100))
#                 loss_list.append(running_loss / 100)
#                 acc_list.append(running_acc / 100)
#                 running_loss = 0.0
#                 running_acc = 0.0
#     print('Finished Training')
#
#     # Plot the loss and accuracy curves
#     plt.plot(loss_list)
#     plt.title('Loss Curve')
#     plt.xlabel('Iterations')
#     plt.ylabel('Loss')
#     plt.show()
#
#     plt.plot(acc_list)
#     plt.title('Accuracy Curve')
#     plt.xlabel('Iterations')
#     plt.ylabel('Accuracy')
#     plt.show()
#
# def test_model(net,testloader,criterion,device):
#     # Test the model
#     correct = 0
#     total = 0
#     test_loss = 0.0
#     with torch.no_grad():
#         for data in testloader:
#             images, labels = data[0].to(device), data[1].to(device)
#             outputs = net(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             test_loss += criterion(outputs, labels).item()
#
#     print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
#     print('Test loss: %.3f' % (test_loss / len(testloader)))
#
#
#
# if __name__ == '__main__':
#     trainloader, testloader=load_data()
#     net, criterion, optimizer, device=get_config()
#     train_model(net,criterion,optimizer,trainloader,device)
#     test_model(net,testloader,criterion,device)
#
#
