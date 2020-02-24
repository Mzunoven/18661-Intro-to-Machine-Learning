# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from numpynet.layer import Dense, ELU, ReLU, SoftmaxCrossEntropy
from numpynet.function import Softmax
from numpynet.utils import Dataloader, one_hot_encoding, load_MNIST, save_csv
from sklearn.metrics import accuracy_score

IntType = np.int64
FloatType = np.float64


class Model(object):
    """Model Your Deep Neural Network
    """

    def __init__(self, input_dim, output_dim):
        """__init__ Constructor

        Arguments:
            input_dim {IntType or int} -- Number of input dimensions
            output_dim {IntType or int} -- Number of classes
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.loss_fn = SoftmaxCrossEntropy(axis=-1)
        self.build_model()
        #self.grad = 0

    def build_model(self):
        """build_model Build the model using numpynet API
        """
        # TODO: Finish this function
        self.layers = [Dense(self.input_dim, 100), ELU(0.9),
                       Dense(100, self.output_dim)]
        #raise NotImplementedError

    def __call__(self, X):
        """__call__ Forward propogation of the model

        Arguments:
            X {np.ndarray} -- Input batch

        Returns:
            np.ndarray -- The output of the model. 
                You can return the logits or probits, 
                which depends on the way how you structure 
                the code.
        """
        # TODO: Finish this function
        for layer in self.layers:
            # print(X.shape)
            X = layer(X)
        return X
        raise NotImplementedError

    def bprop(self, logits, labels, istraining=True):
        """bprop Backward propogation of the model

        Arguments:
            logits {np.ndarray} -- The logits of the model output, 
                which means the pre-softmax output, since you need 
                to pass the logits into SoftmaxCrossEntropy.
            labels {np,ndarray} -- True one-hot lables of the input batch.

        Keyword Arguments:
            istraining {bool} -- If False, only compute the loss. If True, 
                compute the loss first and propagate the gradients through 
                each layer. (default: {True})

        Returns:
            FloatType or float -- The loss of the iteration
        """

        # TODO: Finish this function
        loss, accuracy = self.loss_fn(logits, labels)
        grad = self.loss_fn.bprop()
        for layer in reversed(self.layers):
            grad = layer.bprop(grad)
        return loss, accuracy
        raise NotImplementedError

    def update_parameters(self, lr):
        """update_parameters Update the parameters for each layer.

        Arguments:
            lr {FloatType or float} -- The learning rate
        """
        # TODO: Finish this function
        for layer in reversed(self.layers):
            layer.update(lr)
        #raise NotImplementedError


def train(model,
          train_X,
          train_y,
          val_X,
          val_y,
          max_epochs=100,
          lr=1e-3,
          batch_size=64,
          metric_fn=accuracy_score,
          **kwargs):
    """train Train the model
    Arguments:
        model {Model} -- The Model object
        train_X {np.ndarray} -- Training dataset
        train_y {np.ndarray} -- Training labels
        val_X {np.ndarray} -- Validation dataset
        val_y {np.ndarray} -- Validation labels

    Keyword Arguments:
        max_epochs {IntType or int} -- Maximum training expochs (default: {20})
        lr {FloatType or float} -- Learning rate (default: {1e-3})
        batch_size {IntType or int} -- Size of each mini batch (default: {16})
        metric_fn {function} -- Metric function to measure the performance of 
            the model (default: {accuracy_score})
    """
    # TODO: Finish this function
    np.random.seed(0)
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    train_num = train_X.shape[0]
    val_num = val_X.shape[0]
    for e in range(max_epochs):
        train_loss, train_acc = 0, 0
        val_loss, val_acc = 0, 0
        idxs = np.arange(train_num)
        np.random.shuffle(idxs)
        train_new_x, train_new_y = train_X[idxs], train_y[idxs]
        # training
        for b in range(0, train_num, batch_size):
            range_ = range(b, min(b+batch_size, train_num))
            X = model(train_new_x[range_])
            loss, accuracy = model.bprop(X, train_new_y[range_])
            train_loss += loss
            train_acc += accuracy
            model.update_parameters(lr)
        # validation

        train_loss /= train_num
        train_acc /= train_num
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        for b in range(0, val_num, batch_size):
            range_ = range(b, min(b+batch_size, val_num))
            X = model(val_X[range_])
            loss, accuracy = model.bprop(X, val_y[range_])
            val_loss += loss
            val_acc += accuracy
        val_loss /= val_num
        val_acc /= val_num
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        print("epoch: {}, train acc: {:.2f}%, train loss: {:.3f}, val acc: {:.2f}%, val loss: {:.3f}" .format(
            e+1, train_acc*100, train_loss, val_acc*100, val_loss))
    return train_loss_list, train_acc_list, val_loss_list, val_acc_list
    raise NotImplementedError


def inference(model, X, y=None, batch_size=16, metric_fn=accuracy_score, **kwargs):
    """inference Run the inference on the given dataset

    Arguments:
        model {Model} -- The Neural Network model
        X {np.ndarray} -- The dataset input
        y {np.ndarray} -- The sdataset labels

    Keyword Arguments:
        metric {function} -- Metric function to measure the performance of the model 
            (default: {accuracy_score})

    Returns:
        tuple of (float, float): A tuple of the loss and accuracy
    """

    # TODO: Finish this function
    x_pre = model(X)
    x_max = np.expand_dims(np.max(x_pre, axis=1), axis=1)
    x = np.where(x_pre == x_max)[1]
    return x
    raise NotImplementedError


def main():
    train_X, train_y = load_MNIST(name="train")
    #train_y = load_MNIST(name="train")
    val_X, val_y = load_MNIST(name="val")
    #val_y = load_MNIST(name="val")
    test_X = load_MNIST(name="test")
    #test_loss, test_acc = None, None
    one_hot_train_y = one_hot_encoding(train_y)
    one_hot_val_y = one_hot_encoding(val_y)

    # TODO: 1. Build your model
    model = Model(train_X.shape[1], one_hot_train_y.shape[1])
    model.build_model()
    max_epochs = 150
    # print(val_X.shape, one_hot_val_y.shape)
    # TODO: 2. Train your model with training dataset and
    #       validate it  on the validation dataset
    train_loss_list, train_acc_list, val_loss_list, val_acc_list = train(model, train_X, one_hot_train_y, val_X, one_hot_val_y,
                                                                         max_epochs=150, lr=0.015, batch_size=16)
    # plot loss and accuracy
    x_list = np.arange(1, max_epochs)
    plt.figure(1)
    plt.plot(train_loss_list, label='training')
    plt.plot(val_loss_list, label='validation')
    plt.ylabel('loss', fontsize=16)
    plt.xlabel('num.epochs', fontsize=16)
    plt.legend()
    plt.savefig('loss_curve.png')

    plt.figure(2)
    plt.plot(train_acc_list, label='training')
    plt.plot(val_acc_list, label='validation')
    plt.ylabel('accuracy', fontsize=16)
    plt.xlabel('num.epochs', fontsize=16)
    plt.legend()
    plt.savefig('accuracy_curve.png')
    plt.show
    # TODO: 3. Test your trained model on the test dataset
    #       you need have at least 95% accuracy on the test dataset to receive full scores
    x = inference(model, test_X)
    save_csv(x)
    # Your code starts here

    # Your code ends here
    # print("Test loss: {0}, Test Acc: {1}%".format(test_loss, 100 * test_acc))
    # if test_acc > 0.95:
    #     print("Your model is well-trained.")
    # else:
    #     print("You still need to tune your model")


if __name__ == '__main__':
    main()
