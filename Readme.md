<a name="jI43b"></a>
# 1.Problem Description
It is easy to use the modern neural network framework to train our models but it is hard for learners to get into an idea of what really happen in these API we use.As I have the Introduction to deep learning course,it is an opportunity to get a fully understanding of how forward process and backward process going.<br />In this task,the dataset is SVHN【2】 that is image dataset with 10 classes and the model is designed to recognized  the correct digit on each image.Each image only has 1 digit.<br />The really hard problem is to calculate the forward and backward process of components in neural network in python. Also, the gradient of weights in each layer must be correctly updated.<br />In the training process,the data is loaded by batch so each layer of  the network must consider batchsize dimension.And,the optimizer algorithm can be changed.Not only can it be SGD,also can be Adam.Therefore, the gradient of each layer must be acquired in optimizing. <br />To evaluate the model,it is necessary to test by some indicators.As this is a classfication task，The accuracy must be calculated to show the performance. 
<a name="sKhEe"></a>
# 2.Solution Design
Let's look at the big picture of the solution Design.First,we have `models`package aimed to load data and construct neural network, in which we have `data_module.py`,`nn_module.py`,`net.py`.Next ,we have `optim`package aimed to implement different optimize algorithm.In weights directory,it stores different nets parameters.The `trainer.py`encapsulates all the elements needed in tranning process.The `main.py` specifies the data we need and model we use and start all the processes automatically in the simplest code.The `unit_test.py`stores all the unit classes with its method testing function.<br />In the `models`package  `data_module.py`returns the number of batch size training data or testing data.`nn_module.py` implements the layer including Linear layer,Con2d layer ,Flatten layer,Maxpooling layer and activation function including Relu and Softmax and loss function,with both forward and backward method.The`net.py`implements LNet and MLP,which will be introduced in the Network Architecture section.<br />In the `optim`package,it has `sgd.py`,which literally implements stochastic gradient descent algorithm,with only one parameter learning rate needed to be transfered.In the future,it could have `adam.py`or more optimize algorithms to be chosen.<br />In the  `trainer.py`,it only has a class called Trainer. In this class,it has an important method `fit`,which is need to be transferred two parameters-model,data to get the dataloader and the neural network from `net.py`.Then it will call `fit_epoch` method to actually do the forward computation and calculate the loss and do  backward computation.Finally it will test the network on testing dataset by calliing `eval`method and calculate the accuracy in each epoch for training dataset and test dataset

<a name="cRChU"></a>
# 3.Data Preparetion
The dataset has 73257 training samples and 26032 test samples.Each sample has 3 channels and has 32 ![](https://cdn.nlark.com/yuque/__latex/e0dc12bed73d85d0c6071ab9b5ed4bf3.svg#card=math&code=%5Ctimes&id=xgyOF)32 pixels.For original samples, the shape is like (3,32,32,number of samples).To convenience in forwarding, the samples are transposed into the shape like  (number of samples,32,32,3)<br />The label is a array with shape (number of samples,1) and the value of digit1 has label 1 but the value of digit0 has label 10.Therefore,to match the index,I make the label minus 1 that is the value of digit1 has label 0  and  the value of digit0 has label 9.Then turn the label into one hot encoding,which utimately beacome a array with shape (number of samples,10)<br />To more robustness,the training dataset and test dataset is shuffled respectively in each epoch.It will get the number of batch size data by calling yield.
<a name="IAaoA"></a>
# 4.Network Architecture
In this section,we are going to illustrate what happens in the `net.py`and how our models are constructed.Our first model is a fully connected neeural network,with only one hidden layer Which is a typically multilayer perceptron.It is illustrated like this. 
<a name="rTk8W"></a>
## 4.1MLP
Based on Multilayer perceptron,a fully connected neural network with only one hidden layer is designed to fit these data.<br />The input X with shape of  3 ![](https://cdn.nlark.com/yuque/__latex/e0dc12bed73d85d0c6071ab9b5ed4bf3.svg#card=math&code=%5Ctimes&id=ELtUo) 32![](https://cdn.nlark.com/yuque/__latex/8e26971a7a048ca93d0154f37b01c07c.svg#card=math&code=%5Ctimes%0A&id=yTpJ0)32 has been flatten into a vection with 3072 length.Then it<br />will be linearly mapped into a vector of 1024 length by formula ![image.png](https://cdn.nlark.com/yuque/0/2022/png/25502199/1667789576780-d87147d5-3bc7-4df7-8bae-0accc009b8f7.png#averageHue=%23f5f3f2&clientId=u5902252a-1473-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=35&id=yvFPp&margin=%5Bobject%20Object%5D&name=image.png&originHeight=32&originWidth=146&originalType=binary&ratio=1&rotation=0&showTitle=false&size=1778&status=done&style=none&taskId=u69953fd0-b6c0-4d88-a69d-11fec7830a2&title=&width=159.22222900390625) where W is a matrix.Finally it will be linearly mapped into a vector of 10 length representing the probabilities of 10 classes.<br />The architecture is shown in below.<br /> <br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/25502199/1667737042627-4ccfbed7-0f29-4206-935a-82c1077098cc.png#averageHue=%23e0e0e0&clientId=uf34777c0-c21b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=834&id=qIy0J&margin=%5Bobject%20Object%5D&name=image.png&originHeight=667&originWidth=780&originalType=binary&ratio=1&rotation=0&showTitle=true&size=295922&status=done&style=none&taskId=ucbb96e08-5e15-494e-81d3-74eb8ec1280&title=The%20architecture%20of%20MLP&width=974.9999854713681 "The architecture of MLP")<br />The implement in `net.py` :
```python
class MLP:
    def __init__(self):
        self.model_name = "MLP"
        self.layers =[
            Flatten(),                            # N*3*32*32 -> N*3*1024
            Linear(1024*3,1024),
            ReLu(),
            Linear(1024,10),
        ]


    def forward(self,x):
        out =x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self,delta):
        self.layers.reverse()
        for layer in self.layers :
            delta=layer.backward(delta)
        self.layers.reverse()                        # back to the sequential order
```


<a name="tqMdP"></a>
## 4.2 LNet
Based on LeNet, a net named LNet is devised to apply into this dataset , which just only change the parameters of original LeNet architectures <br />Let's dive into the detail of the architecture of LNet.It has 7 layers not counting the input layer ,which is a 32![](https://cdn.nlark.com/yuque/__latex/8e26971a7a048ca93d0154f37b01c07c.svg#card=math&code=%5Ctimes%0A&id=Kczvr)32 pixel image with 3 channels.In the following,convolution layer 1 has 6 feature maps with   5![](https://cdn.nlark.com/yuque/__latex/8e26971a7a048ca93d0154f37b01c07c.svg#card=math&code=%5Ctimes%0A&id=I12q4)5 kernel  and padding =2,making the output shape remains the same shape as input (i.e. 32![](https://cdn.nlark.com/yuque/__latex/8e26971a7a048ca93d0154f37b01c07c.svg#card=math&code=%5Ctimes%0A&id=oG051)32).<br />Layer 2 is a sub-sampling layer wih 6 feature maps of size of 16 ![](https://cdn.nlark.com/yuque/__latex/e0dc12bed73d85d0c6071ab9b5ed4bf3.svg#card=math&code=%5Ctimes&id=uqgQs)16.Each unit in each feature map is connected to a 2x2 neighborhood in the corresponding feature map in layer 2 and is the maximum of the four in the neighborhood.The outcome will be transferred in to ReLu activation function.<br />Layer 3 and layer 4 are the repetition of the previous two layers.In terms of feature maps, it becomes 16 and the shape of the image becomes to  8![](https://cdn.nlark.com/yuque/__latex/8e26971a7a048ca93d0154f37b01c07c.svg#card=math&code=%5Ctimes%0A&id=st8iz)8.<br />Layer 5 is a flatten layer,which does a reshape operation from  16 ![](https://cdn.nlark.com/yuque/__latex/e0dc12bed73d85d0c6071ab9b5ed4bf3.svg#card=math&code=%5Ctimes&id=T3xRq) 8![](https://cdn.nlark.com/yuque/__latex/8e26971a7a048ca93d0154f37b01c07c.svg#card=math&code=%5Ctimes%0A&id=p2i2z)8 shape into a vector with 1024 length.<br />Layer 6 and Layer 7 are the Linear layer,computing a dot product between the input vector and weight matrix,to which bias is added.The layer 6 is designed to smooth the output in case of directly mapping 1024 to 10 ,which makes the  neural network hard to fit the dataset. <br />The architecture is shown in below.


![image.png](https://cdn.nlark.com/yuque/0/2022/png/25502199/1667788071681-1d79b0a6-6980-48c4-9db3-a88ed2c4e56f.png#averageHue=%23f8f8f8&clientId=u5902252a-1473-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=533&id=u71fca89e&margin=%5Bobject%20Object%5D&name=image.png&originHeight=480&originWidth=1067&originalType=binary&ratio=1&rotation=0&showTitle=true&size=62572&status=done&style=none&taskId=ueba2ea40-cd92-4330-acc0-aa458a8da96&title=fig%20%3Athe%20architecture%20of%20LNet&width=1185.5555869620532 "fig :the architecture of LNet")<br />The implement in `net.py` :
```python
class Lnet:
    def __init__(self):
        self.model_name ="LNet"
        self.layers =[
            Conv2D(3,6,(5,5),stride=1,padding=2), # N*3*32*32 -> N*6*32*32
            MaxPooling(2),                        # N*6*32*32 -> N*6*16*16
            ReLu(),
            Conv2D(6,16,(3,3),stride=1,padding=1), # N*6*16*16 -> N*16*16*16
            MaxPooling(2),                         # N*16*16*16 -> N*16*8*8
            ReLu(),
            Flatten(),                             # N*16*8*8 -> N*1024
            Linear(1024,256),
            ReLu(),
            Linear(256,10)                         # N*10
        ]


    def forward(self,x):
        out =x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self,delta):
        self.layers.reverse()
        for layer in self.layers :
            delta=layer.backward(delta)
        self.layers.reverse()                        # back to the sequential order
```

<a name="honLA"></a>
# 5.Network Trainning
<a name="z6f5U"></a>
## 5.1 Loss function
Now that we have the outcome of a vector of 10 length,we will improve our model performance by  optimizing the CrossEntropy.<br />First we transfer our outcomes into softmax operation to guarantee that each outcome sums up to 1 in the way we believe it is the probability of 10 classes. Also it make sure all the outcomes are positive.<br />Then we calculate the non-negative likelihood(nll) to estimate the similarity of two probability distribution between the softmaxed outcome and the label. We also return the averaged loss value among all examples in the minibatch by callling `mean()`method.<br />Ii is worthy to noted that the input of softmax must have the shape like (N,10),where N = batch size to match the nll loss computation.<br />The implement in `nn_module.py` 
```python
class CrossEntropyLoss:
    """
    The input is expected to contain the unnormalized logits for each class (which do not need to be positive or sum to 1, in general).
    We do softmax and negative-log likelihood together in this loss function
    """
    def __init__(self,reduction =True):
        pass


    def softmax(self,x):
        """
        :param x:shape(batch_size,out_features)
        :return: 
        """
        max_x = np.max(x,axis=1,keepdims=True)
        x_exp = np.exp(x-max_x)
        partition = np.sum(x_exp,axis=1,keepdims=True)
        return  x_exp/partition



    def forward(self,y_hat,y,epsilon=1e-7):
        """
        :param y_hat:  shape(N,C) for each sample ,it has to be a vector of size (C),
        where C =number of classes(i.e.In SVHN dataset C=10).Each value is the probabilities for each class
        :param y:containing class indices,shape(N,classes),for N = batch_size

        :return: negative log likelihood
        """
        self.batch_size = y_hat.shape[0]
        self.target = y
        self.y_hat = self.softmax(y_hat)
        return -np.log( self.y_hat[range(len(y_hat)),np.argmax(y,axis=1)]+epsilon).mean()

    def backward(self):

        delta = (self.y_hat -self.target)/self.batch_size
        return delta
```
<a name="qdZYh"></a>
## 5.2 SGD
Our goal is to minimize the the loss function.To achieve this,we use minibatch SGD.At each step, using a minibatch randomly drawn from our dataset, we estimate the gradient of the loss with respect to the parameters implemented in each module.Next, we update the parameters in the direction that may reduce the loss.<br />The implement in `sgd.py` 
```python
from models import nn_module
class SGD:
    def __init__(self,net,lr=0.01):
        self.lr = lr
        self.layers =net.layers


    def step(self):
        for layer in self.layers:
            if isinstance(layer, nn_module.Linear):
                layer.weight-=self.lr*layer.dw
                layer.bias -= self.lr * layer.db
            if isinstance(layer, nn_module.Conv2D):
                layer.kernel -=self.lr*layer.kernel_grad

```
<a name="Zomdy"></a>
## 5.3 Training
Now we have everything ready,we can train our model.The learning rate is 0.01 and batch size is 250.For LNet,we train 100 epoch and it process 251 examples or so per second. For MLP,we train 100 epoch and it process 3510 examples or so per second.<br />In each epoch, we iterate through the entire training dataset by calling `trainer().fit`method.Then it will call `fit_epoch`method to iterate and show each epoch train accuracy and test accuracy.<br />In each iteration, we grab a minibatch of training examples, and compute the outcome of loss through the module's`forward`method  and  we compute the gradients with respect to each parameter through module's`backward`method . Finally, we will call the optimization algorithm optim by calling`step()` method  to update the model parameters. To visualize the training process,we print the Epoch id ,batch id and average loss of the trained dataset.After a epoch training,we call `eval()`method to compute the accuracy of test dataset.<br />This design is inspired by d2l[3].<br />In summary, we will execute the following loop.
```python
	def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = SGD(model)
        self.cross_entropy = CrossEntropyLoss()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0

        for self.epoch in range(self.max_epochs):
            self.epoch += 1                             # for drawing picture and print condition
            self.fit_epoch()
            if self.epoch%10 ==0:
                print(f"\rEpoch: {self.epoch}/{self.max_epochs}, loss: {self.total_loss[-1]:.3f}",
                      f" train_acc: {self.train_acc[-1]:.3f}",
                      f" test_acc: {self.test_acc[-1]:.3f}",
                      f" examples/sec:{(self.num_train) / (self.time_end - self.time_start):.1f} ",end="\n")
        self.draw()


    def fit_epoch(self):
        self.total_loss_all = 0
        self.train_total_acc = 0
        self.test_total_acc = 0

        self.time_start = time.time()
        for self.train_batch_idx ,(x,y) in enumerate(self.train_dataloader(),start=1):
            output=self.model.forward(x)
            loss =self.cross_entropy.forward(output,y)
            self.model.backward(self.cross_entropy.backward())
            self.optim.step()
            self.total_loss_all+=loss
            self.train_total_acc+=self.accuracy(output,y)

            print(f"\rEpoch: {self.epoch}, train_batch_id: {self.train_batch_idx},"
                  f"loss: {loss/self.train_batch_idx:.3f},train_acc: {self.train_total_acc/self.train_batch_idx:.3f}",
                  flush=True,end="")

        self.time_end = time.time()



        self.train_acc.append(self.train_total_acc/self.train_batch_idx)
        self.total_loss.append(self.total_loss_all/self.train_batch_idx)

        self.eval()
        
```
The evalution function and accuracy function accordingly.
```python
def eval(self):
        for self.val_batch_idx ,(x,y) in enumerate(self.val_dataloader(),start=1):
            output=self.model.forward(x)
            self.test_total_acc+=self.accuracy(output, y)
            print(f"\rEpoch: {self.epoch},test_batch_id : {self.val_batch_idx},  test_acc: {self.test_total_acc/self.val_batch_idx:.3f}", end="",
                  flush=True)
        self.test_acc.append(self.test_total_acc/self.val_batch_idx)

def accuracy(self,y_hat, y):
        """

        :param y_hat:shape(N,C) for each sample ,it has to be a vector of size (C)
        :param y: shape (N,C)  target with one hot encoding
        :return:
        """

        self.each_batch_size = y.shape[0]
        idy_hat = np.argmax(y_hat, axis=1)
        idy = np.argmax(y, axis=1)
        acc = sum(idy == idy_hat) /self.each_batch_size
        return acc
```
<a name="AgL3H"></a>
# 6.Result
In the MLP model,100 iterations through the entire training data has only achieved 35.2% precision  and 30.2 % precision on testing data.The loss has decsended to 1.8 till 40 epoch and fluctuate last 60 epoch.<br />In the LNet model,100 iterations through the entire training data has only achieved 33.2% precision  and 29.2 % precision on testing data.The loss has decsended to2.0 till 20 epoch and fluctuate last 80 epoch.<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/25502199/1667781389526-4bf15c84-230d-485d-bdd9-59d5cf4d4ebe.png#averageHue=%23fcfafa&clientId=u5902252a-1473-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=252&id=u95e1c2cb&margin=%5Bobject%20Object%5D&name=image.png&originHeight=480&originWidth=640&originalType=binary&ratio=1&rotation=0&showTitle=true&size=29729&status=done&style=none&taskId=u8ebeaf49-bbeb-4055-84e6-4517f5c06fb&title=mlp&width=335.8680725097656 "mlp")![image.png](https://cdn.nlark.com/yuque/0/2022/png/25502199/1667799100541-9cac2a65-3281-4854-8134-54fb323bea36.png#averageHue=%23fcfafa&clientId=u4690b366-843b-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=267&id=ua4b11fc4&margin=%5Bobject%20Object%5D&name=image.png&originHeight=480&originWidth=640&originalType=binary&ratio=1&rotation=0&showTitle=true&size=30950&status=done&style=none&taskId=u01e1f748-8514-43fe-92da-be019ec2621&title=Lnet&width=356.1007080078125 "Lnet")
<a name="BPO0U"></a>
# 7.Conclusion
To conclude,it is a really useful exercise to implement the whole deep learning process without using any network framework. By using the SVHN dataset,we design the LNet and MLP model,implementing the basic compoent about convolutional neural network and fully connected neural network.As a result ,the MLP achieve 30.2 % precision on test dataset and LNet achieve 29.2 % precision on test dataset .Although it seems like that they are not very high due to the network architecture design and  training time cost,it is still a great success to get a fully understanding of how network framework works.
<a name="NoeLQ"></a>
# Reference
[1]LeCun Y, Bottou L, Bengio Y, et al. Gradient-based learning applied to document recognition[J]. Proceedings of the IEEE, 1998, 86(11): 2278-2324.<br />[2]Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, Andrew Y. Ng Reading Digits in Natural Images with Unsupervised Feature Learning NIPS Workshop on Deep Learning and Unsupervised Feature Learning 2011<br />[3]Zhang, A., Lipton, Z., Li, M., & Smola, A. (2021). Dive into Deep Learning_. arXiv preprint arXiv:2106.11342_.

<a name="p1XHu"></a>
# Appendix
<a name="P8xNz"></a>
## main.py
```python
from trainer import  Trainer

from models.data_module import SVHNData
from  models.net import  Lnet,MLP


def main(model_name = "MLP"):
    data = SVHNData(root="dataset")
    model = MLP()
    if model_name =="LNet":
        model= Lnet()
    if model_name =="MLP":
        model = MLP()

    trainer =Trainer(max_epochs=50)
    trainer.fit(model,data)
    trainer.save()

if __name__ =="__main__":
    model_name = "LNet"
    main(model_name)
    
```
<a name="DVC59"></a>
## trainer.py
```python
from __future__ import print_function
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np

from models import nn_module
from models.nn_module import CrossEntropyLoss
from optim.sgd import SGD


class Trainer:
    def __init__(self, max_epochs=10,gradient_clip_val=0):
        self.max_epochs =max_epochs


        self.total_loss =[]
        self.train_acc =[]
        self.test_acc = []



    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader
        self.val_dataloader = data.val_dataloader
        self.num_train = data.num_train
        self.num_val = data.num_val


    def prepare_model(self, model):
        self.model = model


    def draw(self):
        x = range(self.epoch)
        plt.clf()
        # plt.ion()
        plt.xlabel('Epoch')
        plt.xlim([0,self.max_epochs])
        plt.plot(x, self.total_loss, color='r',linestyle='-', label='train loss')
        plt.plot(x, self.train_acc, color='b', linestyle='--', label='train acc')
        plt.plot(x, self.test_acc, color='g',  linestyle='--', label='test acc')
        plt.legend()
        plt.show( )
        # plt.pause(1)
        # plt.close()

    def save(self,path ="weights"):
        params = {}
        for index, layer in enumerate(self.model.layers,start=1):
            index =str(index)
            if isinstance(layer, nn_module.Linear):
                params[index+"_linear_weight"] = layer.weight
                params[index+"_linear_bias"] = layer.bias
            if isinstance(layer, nn_module.Conv2D):
                params[index+"_conv2d_kernel"] = layer.kernel
        file_name =path+"/"+self.model.model_name+"params.pkl"
        with open(file_name,'wb') as f:
            pickle.dump(params,f)


    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = SGD(model)
        self.cross_entropy = CrossEntropyLoss()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0

        for self.epoch in range(self.max_epochs):
            self.epoch += 1                             # for drawing picture and print condition
            self.fit_epoch()
            if self.epoch%10 ==0:
                print(f"\rEpoch: {self.epoch}/{self.max_epochs}, loss: {self.total_loss[-1]:.3f}",
                      f" train_acc: {self.train_acc[-1]:.3f}",
                      f" test_acc: {self.test_acc[-1]:.3f}",
                      f" examples/sec:{(self.num_train) / (self.time_end - self.time_start):.1f} ",end="\n")
        self.draw()


    def fit_epoch(self):
        self.total_loss_all = 0
        self.train_total_acc = 0
        self.test_total_acc = 0

        self.time_start = time.time()
        for self.train_batch_idx ,(x,y) in enumerate(self.train_dataloader(),start=1):
            output=self.model.forward(x)
            loss =self.cross_entropy.forward(output,y)
            self.model.backward(self.cross_entropy.backward())
            self.optim.step()
            self.total_loss_all+=loss
            self.train_total_acc+=self.accuracy(output,y)

            print(f"\rEpoch: {self.epoch}, train_batch_id: {self.train_batch_idx},"
                  f"loss: {loss/self.train_batch_idx:.3f},train_acc: {self.train_total_acc/self.train_batch_idx:.3f}",
                  flush=True,end="")

        self.time_end = time.time()



        self.train_acc.append(self.train_total_acc/self.train_batch_idx)
        self.total_loss.append(self.total_loss_all/self.train_batch_idx)

        self.eval()


    def eval(self):
        for self.val_batch_idx ,(x,y) in enumerate(self.val_dataloader(),start=1):
            output=self.model.forward(x)
            self.test_total_acc+=self.accuracy(output, y)
            print(f"\rEpoch: {self.epoch},test_batch_id : {self.val_batch_idx},  test_acc: {self.test_total_acc/self.val_batch_idx:.3f}", end="",
                  flush=True)
        self.test_acc.append(self.test_total_acc/self.val_batch_idx)

    def accuracy(self,y_hat, y):
        """

        :param y_hat:shape(N,C) for each sample ,it has to be a vector of size (C)
        :param y: shape (N,C)  target with one hot encoding
        :return:
        """

        self.each_batch_size = y.shape[0]
        idy_hat = np.argmax(y_hat, axis=1)
        idy = np.argmax(y, axis=1)
        acc = sum(idy == idy_hat) /self.each_batch_size
        return acc
```
<a name="L6gIy"></a>
## models
`models`package aimed to load data and construct neural network, in which we have `data_module.py`,`nn_module.py`,`net.py`.
<a name="RXlCu"></a>
### data_module
```python
import random

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

class DataModule():
    def __init__(self):
      pass

    def get_dataloader(self,train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return  self.get_dataloader(train=False)

class SVHNData(DataModule):
    def __init__(self,batch_size =250,root="../dataset"):
        """
        Load the dataset
        :param batch_size:
        :param root:Where to locate the data
        """
        super().__init__()
        Mat_train =loadmat(root+"/train.mat")
        Mat_test = loadmat(root+"/test.mat")
        train_data, train_labels = Mat_train['X'], Mat_train['y']            # train_data :(32, 32, 3, 73257) train_label :(73257, 1)
        test_data, test_labels = Mat_test['X'], Mat_test['y']                # test_data  :(32, 32, 3, 26032) test_label  :(26032, 1)

        self.num_train = train_data.shape[3]
        self.num_val = test_data.shape[3]

        self.X_train =np.transpose(train_data,(3,2,0,1))
        self.X_test =np.transpose(test_data,(3,2,0,1))

        self.Y_train = self.encode_onehot(train_labels)
        self.Y_test =self.encode_onehot(test_labels)

        self.batch_size =batch_size




    def get_dataloader(self,train=True):
        if train:
            indices = list(range(self.num_train))
            random.shuffle((indices))
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i: min(i + self.batch_size,self.num_train)]
                yield self.X_train[batch_indices], self.Y_train[batch_indices]
        else:
            indices = list(range(self.num_val))
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i: min(i + self.batch_size,self.num_val)]
                yield self.X_test[batch_indices], self.Y_test[batch_indices]


    def encode_onehot(self, y):
        """
        :param y: shape:[num_samples,1] the original label of each image
        :return:
            labels: shape [num_samples,10]
        """
        classes =10
        num_samples = y.shape[0]
        labels = np.zeros(shape=(num_samples, classes))
        for index, y_i in enumerate(y):
            label_value = y_i[0]
            labels[index][label_value-1] = 1

        return labels
```
<a name="KiMNd"></a>
### nn_module
```python
import numpy as np


class Conv2D:
    def __init__(self,in_channel:int, out_channel:int,kernel_size,stride=1,padding=0):
        self.input_channel = in_channel
        self.output_channel =out_channel
        if isinstance(kernel_size, int):
            self.kernel_size_h = self.kernel_size_w = kernel_size
        else:
            assert len(kernel_size) == 2
            self.kernel_size_h = kernel_size[0]
            self.kernel_size_w = kernel_size[1]

        if isinstance(padding, int):
            self.padding_h = self.padding_w = padding
        else:
            assert len(kernel_size) == 2
            self.padding_h = padding[0]
            self.padding_w = padding[1]

        self.stride = stride
        self.padding =padding

        self.kernel = np.random.randn(out_channel,in_channel,self.kernel_size_h,self.kernel_size_w)
        self.bias = np.zeros((1,out_channel))

    def forward(self,x):
        """ Return the result(output_channels,out_h,out_h) of convolution

        The reason why we reshape the weight of kernel is that we want each column represents each channel outcome for sequential listed result of cross correlation.
        For example:
            Column with index zero are ordered by batch_size and the first number of outsize values are produced by the first simple in the batch with the first channels.
            Column with index one  are ordered by batch_size and the first number of outsize values are produced by the first simple in the batch with the second channels
        """

        self.batch_size,self.channels,self.height,self.width = x.shape
        out_h = (self.height + 2 * self.padding_h - self.kernel_size_h) // self.stride + 1
        out_w = (self.width + 2 * self.padding_w - self.kernel_size_w) // self.stride + 1

        if self.padding:
            x=self.pad(x)
        self.x = x

        input_mat = self.imgToMat(x,out_h,out_w)
        kernel_vec = self.kernel.reshape(self.output_channel, -1)
        out = np.dot(input_mat, kernel_vec.T) #+ self.bias

        return out.T.reshape(self.batch_size, out_h, out_w, -1).transpose(0, 3, 1, 2)

    def backward(self,delta):
        """
        Turn the 2D single kernel into 180 degree and do cross correlation operation
        :param delta: delta shape(batch_size,out_channels,height,width)
        :return: delta_next shape(batch_size,in_channels,height,width)
        """
        out_h, out_w = self.height, self.width
        delta_pad = self.pad(delta)

        self.kernel_grad = np.zeros(self.kernel.shape)
        for h in range(self.kernel_size_h):
            for w in range(self.kernel_size_w):
                self.kernel_grad[:, :,h, w] = np.tensordot(delta, self.x[:,:, h:h + out_h, w:w + out_w], ([0,2,3], [0,2,3]))



        input_mat = self.imgToMat(delta_pad,out_h,out_w)
        kernel_vec = self.flip(self.kernel).reshape(self.input_channel, -1)
        delta_next = np.dot(input_mat, kernel_vec.T)

        return delta_next.T.reshape(self.batch_size, out_h, out_w, -1).transpose(0, 3, 1, 2)


    def pad(self,img):
        img = np.pad(img, [(0, 0), (0, 0),
                           (self.padding_h, self.padding_h),
                           (self.padding_w, self.padding_w)],
                     'constant')
        return img

    def imgToMat(self,img,out_h,out_w):
        """ Return the matrix with each row that will be dotted by kernel_weight flatten and number of batch_size*outsize rows

        In the for loop below,we do a slice to get a sub_image with shape (kernel_h,kernel_w) for all channels in the whole batch and reshape it as  (batch_size,channels*sub_image_shape) i.e.sub_image_shape=kernel_h*kernel_w
        As the first number of outsize row in mat will the first sample output ,all the batch output will be insert over number of outsize rows.
        The reason is explained below:
            Row of mat represents the input sub_image of all channels(i.e channels=3 for RGB image) flatten together.
            Each row will be dotted by the weights flatten accordingly as they are both the vector.
            It is batch_size*out_windows_shape of values produced by the cross-correlation operation.
        """
        batch_size,channels,height,width,=img.shape

        mat = np.zeros((self.batch_size * out_h * out_w, channels * self.kernel_size_h * self.kernel_size_w))
        self.outsize = out_w * out_h
        for y in range(out_h):
            y_start = y * self.stride
            y_end = y_start + self.kernel_size_h
            for x in range(out_w):
                x_start = x * self.stride
                x_end = x_start + self.kernel_size_w
                mat[y * out_w + x::self.outsize, :] = img[:, :, y_start:y_end, x_start:x_end].reshape(self.batch_size, -1)
        return mat

    def flip(self,kernel):
        shape =kernel.shape
        dim =len(shape)
        return np.rot90(kernel,k=2,axes=(dim-2,dim-1))

class MaxPooling:
    def __init__(self,size):
        self.size=size

    def forward(self,x):
        self.batch_size, self.channels, self.height, self.width = x.shape
        out = np.zeros((self.batch_size,self.channels,self.height//self.size,self.width//self.size))
        for i in range(0,self.height//self.size):
            for j in range(0, self.width // self.size):
                out[:,:,i,j] =np.max(x[:,:,i:(i+1)*self.size  ,j:(j+1)*self.size],axis=(2,3))
        self.mask = out.repeat(self.size,axis=2).repeat(self.size,axis=3)!=x
        return out

    def backward(self,delta):
        """ Set zero to the position where has no contribution in forwarding
        In this case ,we set that the pooling shape is a square,which means height = size*in_height, width = size*in_width
        :param delta: delta shape(batch_size,channels,in_height,in_width)
        :return: delta_next shape(batch_size,channels,height,width)
        """
        delta_next = delta.repeat(self.size, axis=2).repeat(self.size, axis=3)
        delta_next[self.mask]=0
        return delta_next

class Linear:
    def __init__(self,in_features,out_features,bias=True):
        self.in_features,self.out_features =in_features,out_features
        self.weight = np.random.randn(out_features,in_features)
        self.bias = np.zeros((1,out_features))
        self.is_bias = bias
        self.reset_parameters()

    def reset_parameters(self):
        bound = np.sqrt(6. / (self.in_features + self.out_features))
        self.weight = np.random.uniform(-bound, bound, (self.out_features, self.in_features))


    def forward(self,x):
        """
        :param x: shape(batch_size,in_features)
        :return:
        """
        self.x =x
        return x@self.weight.T+self.bias

    def backward(self,delta):
        """
        :param delta: shape(batch_size,out_features)
        :return: delta_next(batch_size,in_features)
        """
        delta_next = delta@self.weight
        self.dw = delta.T@self.x
        self.db =np.sum(delta,axis=0)

        return delta_next

class ReLu:
    def __init__(self):
        pass
    def  forward(self,x):
        self.mask = (x<=0)
        x[self.mask]=0
        return x


    def backward(self,delta):
        delta[self.mask]=0
        return delta

class Flatten:
    def __init__(self):
        pass

    def forward(self,x):
        self.shape = x.shape
        batch_size = self.shape[0]

        return x.reshape(batch_size,-1)

    def backward(self,delta):
        """
        :param delta: (batch_size,-1)
        :return:
        """
        return delta.reshape(self.shape)


class CrossEntropyLoss:
    """
    The input is expected to contain the unnormalized logits for each class (which do not need to be positive or sum to 1, in general).
    We do softmax and negative-log likelihood together in this loss function
    """
    def __init__(self,reduction =True):
        pass


    def softmax(self,x):
        """
        :param x:shape(batch_size,out_features)
        :return:
        """
        max_x = np.max(x,axis=1,keepdims=True)
        x_exp = np.exp(x-max_x)
        partition = np.sum(x_exp,axis=1,keepdims=True)
        return  x_exp/partition



    def forward(self,y_hat,y,epsilon=1e-7):
        """
        :param y_hat:  shape(N,C) for each sample ,it has to be a vector of size (C),
        where C =number of classes(i.e.In SVHN dataset C=10).Each value is the probabilities for each class
        :param y:containing class indices,shape(N,classes),for N = batch_size

        :return: negative log likelihood
        """
        self.batch_size = y_hat.shape[0]
        self.target = y
        self.y_hat = self.softmax(y_hat)
        return -np.log( self.y_hat[range(len(y_hat)),np.argmax(y,axis=1)]+epsilon).mean()

    def backward(self):

        delta = (self.y_hat -self.target)/self.batch_size
        return delta

```
<a name="tZyGt"></a>
### net
```python
from models.nn_module import Linear,MaxPooling,Conv2D,ReLu,Flatten


class Lnet:
    def __init__(self):
        self.model_name ="LNet"
        self.layers =[
            Conv2D(3,6,(5,5),stride=1,padding=2), # N*3*32*32 -> N*6*32*32
            MaxPooling(2),                        # N*6*32*32 -> N*6*16*16
            ReLu(),
            Conv2D(6,16,(3,3),stride=1,padding=1), # N*6*16*16 -> N*16*16*16
            MaxPooling(2),                         # N*16*16*16 -> N*16*8*8
            ReLu(),
            Flatten(),                             # N*16*8*8 -> N*1024
            Linear(1024,256),
            ReLu(),
            Linear(256,10)                         # N*10
        ]



    def forward(self,x):
        out =x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self,delta):
        self.layers.reverse()
        for layer in self.layers :
            delta=layer.backward(delta)
        self.layers.reverse()                        # back to the sequential order




class MLP:
    def __init__(self):
        self.model_name = "MLP"
        self.layers =[
            Flatten(),                            # N*3*32*32 -> N*1024*3
            Linear(1024*3,1024),
            ReLu(),
            Linear(1024,10),
        ]


    def forward(self,x):
        out =x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self,delta):
        self.layers.reverse()
        for layer in self.layers :
            delta=layer.backward(delta)
        self.layers.reverse()                        # back to the sequential order

```
<a name="xvOtd"></a>
## optim
<a name="gcJZA"></a>
### sgd
```python
from models import nn_module
class SGD:
    def __init__(self,net,lr=0.01):
        self.lr = lr
        self.layers =net.layers


    def step(self):
        for layer in self.layers:
            if isinstance(layer, nn_module.Linear):
                layer.weight-=self.lr*layer.dw
                layer.bias -= self.lr * layer.db
            if isinstance(layer, nn_module.Conv2D):
                layer.kernel -=self.lr*layer.kernel_grad


```
