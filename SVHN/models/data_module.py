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