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

