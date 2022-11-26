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