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
