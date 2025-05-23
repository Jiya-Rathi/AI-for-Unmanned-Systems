import numpy as np
import h5py
import matplotlib.pyplot as plt
from utils import data_loader_mnist, predict_label, DataSplit

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(1)
    
    ###############################################################################################
    # TODO: Use np.random.normal() with mean as 0 and standard deviation as 0.1
    # W Shape (output_dimension, input_dimension), b shape (output_dimension, 1)
    ###############################################################################################
    W1 = np.random.normal(0, 0.1, (n_h, n_x))
    b1 = np.zeros((n_h, 1))
    W2 = np.random.normal(0, 0.1, (n_y, n_h))
    b2 = np.zeros((n_y, 1))
    ### END TODO HERE ###
    
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    ################################################################################
    # TODO: Implement the linear forward pass. Store the result in Z  
    ################################################################################
    Z = np.dot(W, A) + b
    ### END TODO HERE ###
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache

def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    #################################################################################################
    # TODO: Implement the backward pass (i.e., compute the following three terms)
    # dW = ? (the gradient of the mini-batch loss w.r.t. parameter W)
    # db = ? (the gradient of the mini-batch loss w.r.t. parameter b)
    # dA_prev = ? (the gradient of the mini-batch loss w.r.t. X)
    #################################################################################################
    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis = 1, keepdims = True) 
    dA_prev = np.dot(W.T, dZ) 
    ### END TODO HERE ###
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    
    for l in range(1, L+1):
        #################################################################################
        # TODO: Update parameters using the formula:
        # parameter = parameter - learning_rate * gradient
        # And update model parameter W, b for each layer
        #################################################################################
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
        ### END TODO HERE ###
    return parameters

class relu:

    """
        The relu (rectified linear unit) module.

        It is built up with NO arguments.
        It has no parameters to learn.
        self.mask is an attribute of relu. You can use it to store things (computed in the forward pass) for the use in the backward pass.
    """

    def __init__(self):
        self.mask = None

    def forward(self, X):

        """
            The forward pass of the relu (rectified linear unit) module.

            Input:
            - X: A numpy array of arbitrary shape.
            
            Return:
            - forward_output: A numpy array of the same shape of X
        """

        ################################################################################
        # TODO: Implement the relu forward pass. Store the result in forward_output    #
        ################################################################################
        self.mask = (X > 0)
        forward_output = self.mask * X
        ### END TODO HERE ###
        return forward_output

    def backward(self, X, grad):

        """
            The backward pass of the relu (rectified linear unit) module.

            Input:
            - X: A numpy array of arbitrary shape, the input to the forward pass.
            - grad: A numpy array of the same shape of X, where each element is the partial derivative of the mini-batch loss
                 w.r.t. the corresponding element in forward_output.

            Return:
            - backward_output: A numpy array of the same shape as X, where each element is the partial derivative of the mini-batch loss w.r.t. the corresponding element in  X.
        """

        ####################################################################################################
        # TODO: Implement the backward pass
        # You can use the mask created in the forward step.
        ####################################################################################################
        backward_output = grad * self.mask
        ### END TODO HERE ###

        return backward_output

class softmax_cross_entropy:
    def __init__(self):
        self.expand_Y = None
        self.calib_logit = None
        self.sum_exp_calib_logit = None
        self.prob = None

    def forward(self, X, Y):
        self.expand_Y = np.zeros(X.shape).reshape(-1)
        self.expand_Y[Y.astype(int).reshape(-1) + np.arange(X.shape[0]) * X.shape[1]] = 1.0
        self.expand_Y = self.expand_Y.reshape(X.shape)

        self.calib_logit = X - np.amax(X, axis = 1, keepdims = True)
        self.sum_exp_calib_logit = np.sum(np.exp(self.calib_logit), axis = 1, keepdims = True)
        self.prob = np.exp(self.calib_logit) / self.sum_exp_calib_logit

        forward_output = - np.sum(np.multiply(self.expand_Y, self.calib_logit - np.log(self.sum_exp_calib_logit))) / X.shape[0]
        return forward_output

    def backward(self, X, Y):
        backward_output = - (self.expand_Y - self.prob) / X.shape[0]
        return backward_output

if __name__ == "__main__":
    np.random.seed(43) # choose an arbitrary seed

 
    Xtrain, Ytrain, Xtest, Ytest, _, _ = data_loader_mnist(r"C:\\Storage\\SDSU\\Sem 2 Spring 2025\\AI for Unmanned Systems\\Assignment1_DNN\\mnist_subset.json")


    N_train, d = Xtrain.shape
    N_test, _ = Xtest.shape

    train_dataset = DataSplit(Xtrain, Ytrain)
    test_dataset = DataSplit(Xtest, Ytest)


    #################################################################################
    # TODO: reshape one sample input data X and store it in the image
    # Example: reshape Xtrain[[0], :]
    #################################################################################

    ### END TODO HERE ###

    image = Xtrain[0].reshape(28, 28)
    plt.imshow(image)
    plt.show()

    #################################################################################
    # TODO: change the hyperparameters to compare how they impact the final results
    #################################################################################
    num_epoch = 10
    learning_rate = 0.005
    minibatch_size = 5
    ### END TODO HERE ###

    parameters = initialize_parameters(d, 1024, 10)
    grads = {}
    activation = relu()
    entropy = softmax_cross_entropy()
    
    
    cost = []
    for step in range(num_epoch):
        random_idx = np.random.permutation(N_train)
        loss = 0.0
        for i in range(N_train // minibatch_size):
            X, Y = train_dataset.get_example(random_idx[i * minibatch_size : (i + 1) * minibatch_size])

            Z1, cache1 = linear_forward(X.T, parameters["W1"], parameters["b1"])
            ######################################################################################
            # TODO: Call the forward methods of every layer in the model
            # We have given the first and last forward calls
            # Do not modify them.
            ######################################################################################
            A1 = activation.forward(Z1)
            Z2, cache2 = linear_forward(A1, parameters["W2"], parameters["b2"])
            ### END TODO HERE ###

            loss += entropy.forward(Z2.T, Y)

            grad_Z2 = entropy.backward(Z2.T, Y).T
            ######################################################################################
            # TODO: Call the backward methods of every layer in the model in reverse order
            # We have given the first and last backward calls
            # Do not modify them.
            ######################################################################################
            grad_A1, dW2, db2 = linear_backward(grad_Z2, cache2)
            grad_Z1 = activation.backward(Z1, grad_A1)
            grad_X, dW1, db1 = linear_backward(grad_Z1, cache1)
            ### END TODO HERE ###

            grads['dW1'] = dW1
            grads['db1'] = db1
            grads['dW2'] = dW2
            grads['db2'] = db2

            parameters = update_parameters(parameters, grads, learning_rate)

            W1 = parameters["W1"]
            b1 = parameters["b1"]
            W2 = parameters["W2"]
            b2 = parameters["b2"]


            if i % 100 == 99:
                print("The " + str(i+1) + " step")


        cost.append(loss/N_train)


        ### Compute training accuracy
        train_loss = 0
        train_acc = 0
        for i in range(N_train // minibatch_size):
            X, Y = train_dataset.get_example(random_idx[i * minibatch_size : (i + 1) * minibatch_size])

            ############################################################################################################
            # TODO: Call the forward methods of every layer in the model 
            # Then call the predict_label function to predict the label and calculate the accuracy
            ############################################################################################################
            Z1, _ = linear_forward(X.T, parameters["W1"], parameters["b1"])
            A1 = activation.forward(Z1)
            Z2, _ = linear_forward(A1, parameters["W2"], parameters["b2"])
            predictions = predict_label(Z2.T)
            train_acc += np.sum(predictions == Y)
            ### END TODO HERE ###

        train_acc = np.sum(train_acc) / N_train
        print("Training accuracy = ", train_acc)

        ### Compute testing accuracy
        test_loss = 0
        test_acc = 0
        random_idx = np.random.permutation(N_test)
        for i in range(N_test // minibatch_size):
            X, Y = test_dataset.get_example(random_idx[i * minibatch_size : (i + 1) * minibatch_size])

            ############################################################################################################
            # TODO: Call the forward methods of every layer in the model 
            # Then call the predict_label function to predict the label and calculate the accuracy
            ############################################################################################################
            Z1, _ = linear_forward(X.T, parameters["W1"], parameters["b1"])
            A1 = activation.forward(Z1)  
            Z2, _ = linear_forward(A1, parameters["W2"], parameters["b2"])
            predictions = predict_label(Z2.T)
            test_acc += np.sum(predictions == Y)
            ### END TODO HERE ###

        test_acc = np.sum(test_acc) / N_test
        print("Testing accuracy = ", test_acc)
        
    plt.plot(np.arange(len(cost)), np.array(cost))
    plt.xlabel('Number of epochs')
    plt.ylabel('Cost')
    plt.show()

    