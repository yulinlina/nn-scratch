import unittest

import numpy as np

from models.data_module import SVHNData
from models.nn_module import Linear, MaxPooling, Conv2D, CrossEntropyLoss, ReLu, Flatten
from models.net import Lnet
from trainer import Trainer

class MyTestCase(unittest.TestCase):
    def test_linear(self):
        m = Linear(20,30)
        input=np.random.randn(128,20)
        output = m.forward(input)
        assert output.shape==(128,30)
        delta_next = m.backward(output)
        assert delta_next.shape==(128,20)


    def test_conv2(self):
        x_test = np.random.randn(10,3,32,32)
        conv2=Conv2D(3,4,kernel_size = (3,3),padding=1)
        out =conv2.forward(x_test)
        assert (out.shape ==(10,4,32,32))
        delta = conv2.backward(out)
        print(conv2.kernel_grad.shape)
        assert delta.shape==(10,3,32,32)

    def test_maxpooling(self):
        x_test = np.random.randn(10, 3, 32, 32)
        maxpooling =MaxPooling(2)
        out =maxpooling.forward(x_test)
        assert out.shape==(10,3,16,16)
        assert maxpooling.backward(out).shape==(10, 3, 32, 32)


    def test_softmax(self):
        x_test = np.random.randn(2,5)
        x_prob = CrossEntropyLoss().softmax(x_test)
        print(x_prob,np.sum(x_prob,axis=1))

    def test_loss(self):
        y=np.array([[1,0,0],[0,0,1]])
        print(y.shape)
        y_hat = np.array([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
        # y_hat = np.array([[1,2,3,4,5,6,7,8,9,10], [10,9,8,7,6,5,4,3,2,1]])
        cross = CrossEntropyLoss()
        loss =cross.forward(y_hat,y)
        print(loss)
        # print(cross.backward().shape)

    def test_flatten(self):
        x = np.random.randn(10,16,8,8)
        flatten =  Flatten()
        out =flatten .forward(x)
        print(out.shape)
        assert flatten.backward(out).shape == (10,16,8,8)

    def test_relu(self):
        x = np.array([[1,-2],[2,-1]])
        relu = ReLu()
        out = relu.forward(x)
        print(out)
        y =  relu.backward(x)
        print(y)


    def test_net(self):
        x_test = np.random.randn(20,3,32,32)
        out =Lnet().forward(x_test)
        print(out.shape)

    def test_trainer(self):
        model = Lnet()
        data = SVHNData(root="dataset")
        trainer =Trainer(max_epochs=10)
        trainer.fit(model,data)

    def test_acc(self):
        trainer = Trainer(max_epochs=10)
        y = np.array([[1,0,0],[0,1,0],[0,0,1]])
        y_hat = np.array([[0.8,0.2,0.1],[0.8,0.2,0.1],[0.2,0.1,0.8]])
        assert  trainer.accuracy(y_hat,y)==2/3

    def test_dataloader(self):
        data = SVHNData(root="dataset")
        batch = data.train_dataloader()
        print(next(batch)[0].shape)

    def test_flip(self):
        a = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])
        conv2 = Conv2D(3, 4, kernel_size=(3, 3), padding=1)
        out = conv2.flip(a)
        print(out.shape)
        print(out)



if __name__ == '__main__':
    unittest.main()
