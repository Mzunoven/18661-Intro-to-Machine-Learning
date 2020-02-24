# -*- coding: utf-8 -*-

import numpy as np

FloatType = np.float64
IntType = np.int64


class Layer(object):
    """Layer 
    This is the absract class of implementing layer objects
    """

    # DO NOT modify this class

    def __init__(self):
        self.cache = None

    def __call__(self):
        raise NotImplementedError

    def bprop(self):
        raise NotImplementedError


class ReLU(Layer):
    """ReLU Numpy implementation of ReLU activation

    This serves as an exmaple.

    DO NOT modify this class
    """

    def __init__(self):
        """ReLU Constructor
        """
        super(ReLU, self).__init__()

    def __call__(self, x):
        """__call__ Forward propogation through ReLU

        Arguments:
            x {np.ndarray} -- Input of ReLU Layer with shape (B, D)
                B is the batch size, D is the number of dimensions

        Returns:
            np.ndarray -- Output of ReLU Layer
        """
        self.cache = x
        return np.maximum(x, np.zeros_like(x))

    def bprop(self):
        """bprop Backward propogation of ReLU layer

        Returns:
            np.ndarray -- The gradient flowing out of ReLU
        """
        return 1.0 * (self.cache > 0)

    def update(self, lr):
        pass


class Dense(Layer):
    """Dense Numpy implementation of Dense Layer
    """

    def __init__(self, dim_in, dim_out):
        """__init__ Constructor

        Arguments:
            dim_in {int} -- Number of the input dimensions 
            dim_out {int} -- Number of the output dimensions
        """
        super(Dense, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        # The followings are the parameters
        self._W = None
        self._b = None

        # The following are the gradients for the parameters
        self.dW = None
        self.db = None
        self.batch_size = 1
        self.x = np.zeros((self.batch_size, self.dim_in))
        self.y = np.zeros((self.batch_size, self.dim_out))

        # Initialize all the parameters for training
        self._parameter_init()

    def zero_grad(self):
        """zero_grad Clear out the previous gradients
        """
        if self.dW is not None:
            self.dW = np.zeros_like(self.dW)

        if self.db is not None:
            self.db = np.zeros_like(self.db)

    def get_weights(self):
        """get_weights Return the parameters

        Returns:
            list -- A list containing the weights and bias
        """
        return [self._W, self._b]

    def set_weights(self, new_W, new_b):
        """set_weights Set the new parameters

        Arguments:
            new_W {np.ndarray} -- new weights
            new_b {np.ndarray} -- new bias
        """

        self._W = new_W
        self._b = new_b

    def _parameter_init(self):
        """_parameter_init Initialize the parameters
        """
        # TODO: Finish this function
        self._W = np.random.randn(
            self.dim_in, self.dim_out) * np.sqrt(2 / self.dim_in)
        self._b = np.zeros((1, self.dim_out))
        #raise NotImplementedError

    def __call__(self, x):
        """__call__ Forward propogation through Dense layer

        Arguments:
            x {np.ndarray} -- Input of Dense layer with shape (B, D)
                B is the batch size, D is the number of dimensions

        Returns:
            np.ndarray -- Output of Dense Layer
        """

        # TODO: Finish this function
        self.x = x
        self.batch_size = x.shape[0]
        self.y = np.dot(self.x, self._W) + self._b
        return self.y
        raise NotImplementedError

    def bprop(self, grad):
        """bprop Backward propogation of Dense Layer

        Arguments:
            grad {np.ndarray} -- Gradiens comming from the previous layer

        Returns:
            np.ndarray -- The gradient flowing out of Dense Layer
        """
        # TODO: Finish this function
        self.dW = np.dot(self.x.T, grad)
        self.db = np.sum(grad, axis=0) / grad.shape[0]
        return np.dot(grad, self._W.T)
        raise NotImplementedError

    def update(self, lr):
        """update Update the parameters 

        Arguments:
            lr {FloatType or float} -- learning rate
        """
        self._W = self._W - lr * self.dW
        self._b = self._b - lr * self.db
        #raise NotImplementedError


class ELU(Layer):
    """ELU Numpy implementation of ELU activation
    """

    def __init__(self, alpha):
        """ELU Constructor
        """
        super(ELU, self).__init__()
        self.alpha = alpha
        self.batch_size = 16
        self.features = 100
        self.x = np.zeros((self.batch_size, self.features))
        self.y = np.zeros((self.batch_size, self.features))
        self.loss = np.zeros((self.batch_size, self.features))
        self.lr = 0

    def __call__(self, x):
        """__call__ Forward propogation through ReLU

        Arguments:
            x {np.ndarray} -- Input of ELU Layer with shape (B, D)
                B is the batch size, D is the number of dimensions

        Returns:
            np.ndarray -- Output of ELU Layer
        """

        # TODO: Finish this functions
        self.x = x
        self.batch_size = self.x.shape[0]
        self.features = self.x.shape[1]
        self.y = (abs(self.x)+self.x) / 2
        # for i in range(self.batch_size):
        #     for j in range(self.features):
        #         if self.x[i, j] > 0:
        #             self.y[i, j] = self.x[i, j]
        #         else:
        #             self.y[i, j] = self.alpha * (np.exp(self.x[i, j]) - 1)
        return self.y
        raise NotImplementedError

    def bprop(self, grad):
        """bprop Backward propogation of ELU layer

        Returns:
            np.ndarray -- The gradient flowing out of ELU
        """

        # TODO: Finish this function
        self.loss = self.x
        # self.loss[self.loss > 0] = 1
        # self.loss[self.loss <= 0] = self.alpha * \
        #     np.exp(self.loss[self.loss <= 0])
        self.loss[self.loss > 0] = 1
        self.loss[self.loss <= 0] = 0
        # for i in range(self.loss.shape[0]):
        #     for j in range(self.loss.shape[1]):
        #         if self.loss[i, j] > 0:
        #             self.loss[i, j] = 1
        #         else:
        #             self.loss[i, j] = self.alpha * np.exp(self.loss[i, j])
        return grad * self.loss
        raise NotImplementedError

    def update(self, lr):
        self.lr = lr
        #raise NotImplementedError


class SoftmaxCrossEntropy(Layer):
    """SoftmaxCrossEntropy Numpy implementation of Softmax and Cross Entroppy 
    """

    def __init__(self, axis=-1):
        """__init__ Constructor

        Keyword Arguments:
            axis {int} -- The axis on which to apply the Softmax (default: {-1})
        """
        super(SoftmaxCrossEntropy, self).__init__()
        self.axis = axis
        self.batch_size = 16
        self.classes = 10
        self.x = np.zeros((self.batch_size, self.classes))
        self.y = np.zeros((self.batch_size, self.classes))
        self.loss = 0
        self.acc = 0
        self.x_max = np.zeros((self.batch_size, 1))
        self.true_class = np.zeros((self.batch_size, 1))
        self.exp_norm = np.zeros((self.batch_size, self.classes))
        self.softmax_x = np.zeros((self.batch_size, self.classes))

    def __call__(self, logits, labels):
        """__call__ Forward propogation through Softmax

        Arguments:
            logits {np.ndarray} -- Input logits with shape (B, C)
                B is the batch size, D is the number of classes
            labels {np.ndarray} -- Input one-hot encoded labels with shape (B, C)
                B is the batch size, D is the number of classes

        Returns:
            FloatType --  loss and accuracy per batch
        """

        # TODO: Finish this function
        self.batch_size = logits.shape[0]
        self.classes = logits.shape[1]
        self.true_class = np.zeros((self.batch_size, 1))
        self.x = logits
        self.y = labels
        self.x_max = np.max(self.x, axis=1)
        self.exp_norm = np.exp(self.x - np.expand_dims(self.x_max, axis=1))
        self.softmax_x = self.exp_norm / \
            np.expand_dims(np.sum(self.exp_norm, axis=1), axis=1)
        for i in range(self.batch_size):
            self.true_class[i] = - \
                np.log(np.dot(self.y[i, :], self.softmax_x[i, :].T))
        self.loss = np.sum(self.true_class)
        #print(np.where(self.x == np.expand_dims(self.x_max, axis=1)))
        self.acc = np.sum(np.where(self.x == np.expand_dims(self.x_max, axis=1))[
                          1] == np.where(self.y == 1)[1])
        return (self.loss, self.acc)
        raise NotImplementedError

    def bprop(self):
        """bprop Backward propogation of Softmax layer

        Returns:
            np.ndarray -- The gradient flowing out of SoftmaxCrossEntropy

        Raises:
            NotImplementedError: [description]
        """
        # TODO: Finish this function
        return self.softmax_x - self.y
        raise NotImplementedError
