import numpy as np
import json




def data_loader_mnist(dataset):
    # This function reads the MNIST data and separate it into train, val, and test set
    with open(dataset, 'r') as f:
        data_set = json.load(f)
    train_set, valid_set, test_set = data_set['train'], data_set['valid'], data_set['test']

    Xtrain = np.array(train_set[0])
    Ytrain = np.array(train_set[1])
    Xvalid = np.array(valid_set[0])
    Yvalid = np.array(valid_set[1])
    Xtest = np.array(test_set[0])
    Ytest = np.array(test_set[1])

    return Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest

class DataSplit:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.N, self.d = self.X.shape

    def get_example(self, idx):
        batchX = np.zeros((len(idx), self.d))
        batchY = np.zeros((len(idx), 1))
        for i in range(len(idx)):
            batchX[i] = self.X[idx[i]]
            batchY[i, :] = self.Y[idx[i]]
        return batchX, batchY

def predict_label(f):
    # This is a function to determine the predicted label given scores
    if f.shape[1] == 1:
        return (f > 0).astype(float)
    else:
        return np.argmax(f, axis=1).astype(float).reshape((f.shape[0], -1))

if __name__ == "__main__":
    Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest = data_loader_mnist("./mnist_subset.json")

    print(Xtrain.shape, Ytrain.shape)