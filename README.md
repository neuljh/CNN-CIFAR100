# CNN-CIFAR100
# CNN模型，对CIFAR100数据集进行分类，三连球球啦！

(1)Tensorflow库构建CNN模型(下文简称CNN_tensorflow)
1)导入必要的库。题目要求使用Tensorflow库构建CNN模型。这里使用了 TensorFlow 和 Keras 库，以及 numpy 和 matplotlib 库用于数据处理和可视化。
```python
import tensorflow as tf 
from tensorflow import keras
import numpy as np 
from keras.datasets import cifar100
import matplotlib.pyplot as plt
```
2)加载 CIFAR-100 数据集
```python
(x_train, y_train),(x_test, y_test) = cifar100.load_data(label_mode='fine')
```
这里使用了 Keras 提供的 CIFAR-100 数据集加载函数。
在 cifar100.load_data() 函数中，label_mode 参数用于控制标签的加载方式。CIFAR-100 数据集中有两种标签，一种是精细标签（fine labels），共有100个类别，另一种是粗略标签（coarse labels），共有20个类别，每个粗略类别下面包含5个精细类别。因此，label_mode 参数可以取两个值：'fine' 和 'coarse'。
当 label_mode='fine' 时，load_data() 函数将加载精细标签，返回的训练集标签和测试集标签是包含类别序号（0-99）的一维数组。当 label_mode='coarse' 时，load_data() 函数将加载粗略标签，返回的训练集标签和测试集标签是包含粗略类别序号（0-19）的一维数组。
在上述代码中，label_mode='fine' 表示加载精细标签，即返回一个包含0-99之间整数的一维数组，每个整数代表一个类别。这样可以更准确地识别物体，因为共有100个不同的类别。

3)将训练集和测试集的像素值类型从整数转换为单精度浮点数类型。将像素值归一化到 [0, 1] 的范围内。使用了 Numpy 库提供的 astype() 函数来进行数据类型转换。
```python
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train / 255.0
x_test = x_test / 255.0
```
这是一个常见的数据预处理步骤，通常是为了将像素值归一化到 [0, 1] 的范围内，以便更好地进行模型训练。浮点数比整数更适合进行数学计算,在深度学习中，通常使用浮点数来存储数据。

4)定义卷积神经网络模型，包含两个卷积层、一个池化层、一个全连接层和一个 softmax 层。这里使用了 Keras 提供的 Sequential 模型，将各个层按顺序添加到模型中。
```python
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), strides=1, activation='relu', input_shape=(32, 32, 3)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),                           
    keras.layers.MaxPooling2D((2, 2)),                                           
    keras.layers.Dropout(0.5),                                                   
    keras.layers.Flatten(),                                                       
    keras.layers.Dense(128, activation='relu'),                                  
    keras.layers.Dropout(0.5),                                                   
    keras.layers.Dense(100, activation='softmax')                                  
])
```
在上述示例代码所构建地CNN模型中，各个层的功能如下：
Conv2D层：它有32个3x3大小的过滤器，步幅为1，激活函数为ReLU，输入形状为（32,32,3），表示输入数据是RGB图像。
Conv2D层：它有64个3x3大小的过滤器，激活函数为ReLU。这个卷积层的作用是将特征进一步提取出来。
MaxPooling2D层：使用2x2的池化窗口，从卷积层输出中提取最大值。它将卷积层的输出减小一半，减少了模型中的参数数量和计算负担。这里的大小为（2,2），表示每个2x2的像素块被压缩为单个像素，其值为所包含像素的最大值。
Dropout层：有助于减轻过拟合。dropout 可以将一定比例的输入单元随机设置为0，从而防止过度依赖于特定的输入单元。这里随机将50%的输入单元设置为0。
Flatten层：将卷积层输出的多维张量展平为一维张量，用于传递给全连接层。这是因为完全连接的层需要一维输入。
Dense层：包含128个神经元，应用ReLU激活函数。这个层接收来自之前的卷积和池化层的特征，并使用它们来学习数据的高级表示。
Dropout层：再次进行防止过拟合，随机将50%的输入单元设置为0。
Dense层：具有100个神经元和softmax激活函数。这是我们的输出层，用于分类任务。这里是100个类别的分类任务，因此我们有100个神经元，每个神经元代表一个类别。softmax 函数将每个神经元的输出转换为介于0和1之间的概率分布，以便可以解释为每个类别的预测概率。
下图是系统生成的模型结构图（含DROP_OUT层）和参数数量：

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/3ed025a4-cd44-42af-a61a-42532025a583)

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/dcf2c3bb-19f7-48f0-8468-0dfabb342b54)


5)编译模型，定义损失函数和优化器。
```python
  model.compile(optimizer=keras.optimizers.Adam(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```
model.compile 是 Keras 中用来编译模型的函数，它有三个重要的参数：
optimizer: 优化器用来根据损失函数来更新模型的权重参数，使得模型的损失函数值不断降低，是用来指定训练过程中使用的优化算法，如 SGD、Adam 等。在本实例代码中，keras.optimizers.Adam() 表示使用 Adam 优化器，它是一种常用的随机梯度下降法变体，可以自适应地调整每个参数的学习率，以便更有效地优化模型。
loss: 损失函数，用来衡量模型的预测值与真实值之间的差异，作为优化算法的目标函数。在本示例代码中，sparse_categorical_crossentropy 是一种交叉熵损失函数，它在多分类问题中经常被使用。sparse_categorical_crossentropy 是一个分类问题的损失函数，适用于类别标签为整数编码的情况。
metrics: 评价指标，用来衡量模型的性能，如准确率、精确率、召回率等。在本例中，metrics=['accuracy'] 表示我们希望用准确率来衡量模型的性能。准确率指模型预测正确的样本数与总样本数的比例。因此，模型在训练过程中会输出每个 epoch 结束时的训练集和测试集准确率，以便我们可以了解模型的训练情况。

6)训练模型，使用 fit() 函数进行模型训练。这里设置了 10 个 epoch，即模型会对训练数据进行 10 次迭代训练。
```python
history = model.fit(x_train, y_train, epochs=10)
```
7)在测试集上评估模型，使用 evaluate() 函数计算模型在测试集上的损失和准确率。
```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test acc: %f' % test_acc)
```
(2)CNN_tensorflow实验结果

(1)直接输出结果

不加入DROP_OUT层:

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/5fdc28ba-a87d-43d3-a97c-362f14eebf7f)

加入DROP_OUT层:

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/42d9b44d-071d-4b99-9bc0-b66ebb3eb9d3)

(2)CNN_tensorflow在加入和不加入DROP_OUT层情况下的训练集上的loss和accuracy：

不加入DROP_OUT层:

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/760543a8-1c5b-4b3e-a661-3fc31f6df354)


加入DROP_OUT层:

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/686ce1c4-afa0-4ffe-a67a-4682a6fdce96)

发现加入DROP_OUT层之后，在训练集上的训练效果变差了，loss远大于不加入DROP_OUT层的情况；accuracy远小于不加入DROP_OUT层的情况，并且数值仅仅达到前者一半左右。

(3)CNN_tensorflow在加入和不加入DROP_OUT层情况下的测试集上的loss和accuracy：

不加入DROP_OUT层:

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/6a0a9f41-0d52-4f0b-871b-fced647fcef4)

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/c9632e70-d100-4d48-81e0-02e45ed2979a)

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/fa7fa8e4-e918-4889-b343-997a311ea8bd)


加入DROP_OUT层:

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/80c5fb87-a1c8-4357-a265-95baadc88f65)

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/019fd9b0-89eb-4d8c-a485-b87573dd1040)

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/bcf448e6-eab9-451f-ab09-c5f3dfa3dcdd)


	发现加入DROP_OUT层之后，尽管在训练集上的训练效果变差了，但是测试集上的训练效果和补加入DROP_OUT层的情况相比，具有明显的提高。我推测应该是在模型在训练过程中虽然没有学习成熟所给特征，但是所学习的特征都能很好的衡量对图片的分类。为了验证我的猜想，下面又在加入DROP_OUT层的情况下多运行了几次代码：

 (3)Pytorch库构建CNN模型(下文简称CNN_pytorch)
 
1)导入必要的库。这里采用Pytorch库构建CNN模型，因此引入Pytorch相关的库。

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/31b6662d-4735-4ada-887b-1e26f16b6c1c)

2)定义CNN模型。

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/050d46dd-663e-4870-9512-f8bf75e9afe4)

定义了一个名为CNN的类，它是PyTorch中nn.Module类的子类，表示这是一个神经网络模型。该类具有两个主要方法：初始化方法和前向传播方法。
初始化方法包含4个层级：2个卷积层，1个全连接层和1个输出层，这些层级在该网络中实现了图像分类任务的必要功能。其中卷积层使用了32个3x3的卷积核和64个3x3的卷积核。在初始化方法中，使用nn.Conv2d()函数定义了这些卷积层，并使用self.conv1和self.conv2作为层的名称。
全连接层定义了两个层级，分别包含512个神经元和100个神经元。同样，使用nn.Linear()函数定义了这些层级，并使用self.fc1和self.fc2作为层的名称。
前向传播方法是神经网络模型的核心，该方法接受输入x，并对输入进行一系列计算以生成输出。具体而言，前向传播方法首先将输入x传递给第一层卷积层，然后使用ReLU激活函数进行非线性变换。接下来，对卷积结果进行2x2最大池化，该池化操作将卷积层的输出向下采样，从而降低输出维度。然后，将结果传递给第二层卷积层，并再次使用ReLU激活函数进行非线性变换，最后再进行一次2x2最大池化操作。
在这之后，通过x.view()函数将输出形状改为一个向量，并将该向量传递给全连接层。同样，这里使用ReLU激活函数进行非线性变换，并将结果传递给输出层，输出层的输出是一个大小为100的向量，表示图像属于100个类别中的哪一个。最后，该方法返回输出向量。

3)加载CIFAR100数据集并定义数据加载器

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/281e9f59-5cfc-4470-8d5a-a1ad6b2d7a55)

4)加载训练基本信息

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/a0ab1633-5d5e-4d49-b307-a7e43c56a859)

首先创建了一个 CNN 类的实例，这个类是之前定义的卷积神经网络模型。然后定义了一个交叉熵损失函数，这个函数常用于分类问题中计算损失。接下来定义了一个随机梯度下降（SGD）优化器，它将用于更新模型的参数，其中 lr 参数指定了学习率，momentum 参数指定了动量。最后，函数检查是否有可用的 GPU，如果有，则使用第一个可用的 GPU，否则使用 CPU。
最终返回了一个包含四个元素的元组 (net, criterion, optimizer, device)，分别表示神经网络、损失函数、优化器和设备信息。这些元素将在训练模型时使用。
5)训练模型
```python
def train_model(net,criterion,optimizer,trainloader,device):
    # Train the model
    net.to(device)
    for epoch in range(10):  # loop over the dataset multiple times
        print("Epoch[{}/{}]:".format(epoch + 1, 10))
        running_loss = 0.0
        running_acc = 0.0
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
            running_loss += loss.item()
            running_acc += acc.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f, accuracy: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100, running_acc / 100))
                running_loss = 0.0
                running_acc = 0.0
    print('Finished Training')
```

6)测试模型
```python
def test_model(net,testloader,criterion,device):
    # Test the model
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss += criterion(outputs, labels).item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    print('Test loss: %.3f' % (test_loss / len(testloader)))
```
(4)CNN_pytorch实验结果

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/55b8409f-f92e-4daf-9409-2b470b749d40)

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/eb45f719-614e-41d5-bd17-742c5b8f3f24)

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/f7f3e9d0-38f5-4bd7-a1ee-2a32d1569305)

可视化结果：

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/2110fb02-2cfc-4e6b-954d-9209af11b636)

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/21e86be9-05d6-4d3c-aa4b-e59163b0b52c)

最后测试集和训练集的准确率均在15-16%左右。

(5)模型的优化

1)CNN模型的构建

①　优化尝试1

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/c49d4d00-cbe9-45e2-a073-3800f8f73fa0)

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/d27551d9-8e9f-4566-bee1-5f2d01b1ee23)

②　优化尝试2

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/bf7f00e8-0305-451d-b7a5-fd31fe2ddf9e)

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/14e99cb6-6e72-4dc8-9b89-a3c9f8461f28)


2)实验结果
①　优化尝试1

在测试集的准确度达到了53%。

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/d7fed9ff-6f87-4ace-8e5c-ab679ae40e65)

②　优化尝试2

在测试集的准确度达到了59%。

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/78e9596e-7d12-4fab-8df2-4dfd40511d58)

3)实验结果可视化
①　优化尝试1
训练准确率(Train accuracy)和训练损失值(Train loss)随模型迭代次数的变化趋势图：

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/6f597898-d1cc-4399-9d07-3cb80ac9a99c)

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/2f90848f-9448-4ee9-a2a8-82476f8afd70)

训练准确率(Train accuracy)和验证准确率(Validation accuracy)随训练轮数(epoch)的变化趋势图；训练损失值(Train loss)和验证损失值(Validation loss)随训练轮数(epoch)的变化趋势图：

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/73d16d91-4af1-4d69-b240-10621caae8f3)

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/2661d4dc-d35c-478e-8034-f8e17d190139)

在10k测试集上测试得到模型的最终准确率为53%，随机抽取5*5的可视化结果图，结果显示如图：

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/97b91a48-cc14-4a7d-9297-f48ba39801d2)

说明我们的参数调整方向是正确的，模型此时训练效果不错，但是仍然存在较大的进步空间。

②　优化尝试2
训练准确率(Train accuracy)和训练损失值(Train loss)随模型迭代次数的变化趋势图：

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/8e660860-5b79-449a-b8ca-8baa82750394)

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/7c87cabe-9bd1-455a-8de7-d21f8aeb4340)

训练准确率(Train accuracy)和验证准确率(Validation accuracy)随训练轮数(epoch)的变化趋势图；训练损失值(Train loss)和验证损失值(Validation loss)随训练轮数(epoch)的变化趋势图：

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/06e92fcd-cfcc-49a4-a508-082e694c9057)

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/1a74f550-9b7d-43d7-8ee8-18ccedee03a1)

在10k测试集上测试得到模型的最终准确率为59%，从53%上升到了59%。随机抽取5*5的可视化结果图，结果显示如图：

![image](https://github.com/neuljh/CNN-CIFAR100/assets/132900799/be1e3b2a-786f-4ff2-b6bf-415fec342779)

说明我们的参数调整方向是正确的，模型此时训练效果不错，但是仍然存在较大的进步空间。
