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



