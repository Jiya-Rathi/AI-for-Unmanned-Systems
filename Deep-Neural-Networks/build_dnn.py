import numpy as np
import h5py
import matplotlib.pyplot as plt
from utils import data_loader_mnist, predict_label, DataSplit
from build_nn import linear_forward, linear_backward, relu, softmax_cross_entropy

def initialize_parameters_deep(layer_dims):
    """
    Initialize parameters for a deep network using np.random.normal() with mean 0 and standard deviation 0.1.
    """
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)  # number of layers (including input)
    
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.normal(0, 0.1, (layer_dims[l], layer_dims[l-1]))
        parameters['b' + str(l)] = np.random.normal(0, 0.1, (layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        
    return parameters

class dropout:
    def __init__(self, r):
        self.r = r
        self.mask = None

    def forward(self, X, is_train):
        """
        Forward pass for dropout.
        """
        if is_train:
            self.mask = (np.random.uniform(0.0, 1.0, X.shape) >= self.r).astype(float) * (1.0 / (1.0 - self.r))
        else:
            self.mask = np.ones(X.shape)
        forward_output = np.multiply(X, self.mask)
        return forward_output

    def backward(self, X, grad):
        """
        Backward pass for dropout.
        """
        backward_output = grad * self.mask
        return backward_output

class L_layer_model:
    def __init__(self, layer_dims, isTrain):
        self.params = initialize_parameters_deep(layer_dims)
        self.L = len(layer_dims)
        self.isTrain = isTrain
        self.nonlinear = dict()
        # For each layer (except input), create relu and dropout modules.
        for l in range(1, self.L):
            self.nonlinear['relu' + str(l)] = relu()
            self.nonlinear['drop' + str(l)] = dropout(0.5)  # dropout rate set to 0.5
        self.nonlinear['softmax'] = softmax_cross_entropy()
        self.caches = dict()
        self.grads = dict()
        self.learning_rate = 0.01

    def forward(self, X):
        """
        Implements the forward propagation for the deep network.
        For layers 1 to L-1, perform linear forward then relu activation and dropout.
        For the last layer, perform only linear forward.
        """
        input_X = X
        for l in range(1, self.L):
            Z, linear_cache = linear_forward(input_X, self.params["W" + str(l)], self.params["b" + str(l)])
            self.caches["L" + str(l)] = linear_cache
            if l < self.L - 1:
                # Save the linear output Z for the relu backward.
                A = self.nonlinear["relu" + str(l)].forward(Z)
                D = self.nonlinear["drop" + str(l)].forward(A, self.isTrain)
                self.caches["relu" + str(l)] = Z  # store pre-activation value for relu
                self.caches["drop" + str(l)] = D
                input_X = D
            else:
                input_X = Z
        return input_X

    def backward(self, Y, loss_grad):
        """
        Implements the backward propagation for the deep network.
        """
        # Backward for the last (output) layer.
        linear_grad, dW_last, db_last = linear_backward(loss_grad, self.caches["L" + str(self.L - 1)])
        self.grads["dW" + str(self.L - 1)] = dW_last
        self.grads["db" + str(self.L - 1)] = db_last
        grad_next = linear_grad
        
        # Loop over layers L-1 to 1 (in reverse order)
        for l in range(2, self.L):
            curr = self.L - l  # current layer index
            # Backward pass for dropout layer.
            d_drop = self.nonlinear["drop" + str(curr)].backward(self.caches["relu" + str(curr)], grad_next)
            # Backward pass for relu activation.
            d_relu = self.nonlinear["relu" + str(curr)].backward(self.caches["relu" + str(curr)], d_drop)
            linear_grad, dW_curr, db_curr = linear_backward(d_relu, self.caches["L" + str(curr)])
            self.grads["dW" + str(curr)] = dW_curr
            self.grads["db" + str(curr)] = db_curr
            self.grads["drop" + str(curr)] = d_drop
            self.grads["relu" + str(curr)] = d_relu
            grad_next = linear_grad
        
        if self.isTrain: 
            self.L_layer_update_parameters(self.params, self.grads, self.learning_rate)
        
        return self.grads
    
    def L_layer_update_parameters(self, parameters, grads, learning_rate):
        """
        Update all parameters using gradient descent.
        """
        for l in range(1, self.L):
            parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
            parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]
        self.params = parameters

if __name__ == "__main__":
    np.random.seed(43)
    # Load the MNIST subset.
    Xtrain, Ytrain, Xtest, Ytest, _, _ = data_loader_mnist("C:\\Storage\\SDSU\\Sem 2 Spring 2025\\AI for Unmanned Systems\\Assignment1_DNN\\mnist_subset.json")
    
    #################################################################################
    # TODO: reshape one sample input data X and store it in the image
    #################################################################################
    image = Xtrain[0].reshape(28, 28)
    #################################################################################
    # END TODO
    #################################################################################
    
    #plt.imshow(image)
    #plt.show()
    
    #################################################################################
    # TODO: change the hyperparameters to compare how they impact the final results
    #################################################################################
    num_epoch = 100
    learning_rate = 0.05
    dropoff_rate = 0.5
    layer_dims = [Xtrain.shape[1], 1024, 512,256,128, 10]
    minibatch_size = 5
    #################################################################################
    # END TODO
    #################################################################################
    
    train_dataset = DataSplit(Xtrain, Ytrain)
    test_dataset = DataSplit(Xtest, Ytest)
    
    model = L_layer_model(layer_dims, True)
    model.learning_rate = learning_rate  # update learning rate from hyperparameter
    entropy = softmax_cross_entropy()
    cost = []
    N_train = Xtrain.shape[0]
    N_test = Xtest.shape[0]
    
    for step in range(num_epoch):
        random_idx = np.random.permutation(N_train)
        loss = 0.0
        
        for i in range(N_train // minibatch_size):
            X_batch, Y_batch = train_dataset.get_example(random_idx[i * minibatch_size : (i + 1) * minibatch_size])
            # Forward pass: transpose input so that its shape is (features, examples)
            Z = model.forward(X_batch.T)
            loss += entropy.forward(Z.T, Y_batch)
            grad_loss = entropy.backward(Z.T, Y_batch).T
            model.backward(Y_batch, grad_loss)
            if i % 100 == 99:
                print("Epoch", step+1, "Mini-batch", i+1, "loss:", loss)

        cost.append(loss / N_train)
        
        # Compute training accuracy.
        train_correct = 0
        for i in range(N_train // minibatch_size):
            X_batch, Y_batch = train_dataset.get_example(random_idx[i * minibatch_size : (i + 1) * minibatch_size])
            Z = model.forward(X_batch.T)
            predictions = predict_label(Z.T)
            train_correct += np.sum(predictions == Y_batch)
        train_acc = train_correct / N_train
        print("Epoch", step+1, "Training accuracy =", train_acc)
        
        # Compute testing accuracy.
        test_correct = 0
        random_idx_test = np.random.permutation(N_test)
        for i in range(N_test // minibatch_size):
            X_batch, Y_batch = test_dataset.get_example(random_idx_test[i * minibatch_size : (i + 1) * minibatch_size])
            Z = model.forward(X_batch.T)
            predictions = predict_label(Z.T)
            test_correct += np.sum(predictions == Y_batch)
        test_acc = test_correct / N_test
        print("Epoch", step+1, "Testing accuracy =", test_acc)
    
    plt.plot(np.arange(len(cost)), np.array(cost))
    plt.xlabel('Number of epochs')
    plt.ylabel('Cost')
    plt.show()
    print("Final training accuracy = ", train_acc)
    print("Final testing accuracy = ", test_acc)