import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def cost_SVM(W, X, Y):
    #Hinge loss
    N = X.shape[0]
    distances = 1 - Y * (np.dot(X, W))
    distances[distances < 0] = 0
    hinge_loss = C * (np.sum(distances) / N)
    #Cost
    cost = 1 / 2 * np.dot(W, W) + hinge_loss
    return cost

def calculate_gradient(W, X, Y):
    distance = 1 - (Y * np.dot(X, W))
    dw = np.zeros(len(W))

    for i, d in enumerate(distance):
        if max(0, d) == 0:
            di = W
        else:
            di = W - (C * Y[i] * X[i])
        dw += di

    dw = dw / len(Y)
    return dw


def sgd(X, Y):
    max_epochs = 5000
    w = np.zeros(X.shape[1])
    n = 0
    prev_cost = float("inf")
    #Stopping Criteria
    cost_threshold = 0.01

    #Gradient Descent
    for epoch in range(1, max_epochs):
        descent = calculate_gradient(w, X, Y)
        w = w - (learning_rate * descent)
        #Convergence Check
        if epoch == 2 ** n or epoch == max_epochs - 1:
            cost = cost_SVM(w, X, Y)
            print("Epoch is: {} and Cost is: {}".format(epoch, cost))
            #Stoppage criteria
            if abs(prev_cost - cost) < cost_threshold * prev_cost:
                return w
            prev_cost = cost
            n += 1
    
    return w

if __name__ == '__main__':
    #Model HyperParameters
    C = 10000
    learning_rate = 0.1

    #Read Data
    data = pd.read_csv('iris.csv')
    print(data)
    
    Y = data.loc[:, 'class']
    X = data.iloc[:, :-1]

    #Scaling Data and adding a feature to each row for 'b'
    X_scaled = MinMaxScaler().fit_transform(X.values)
    X = pd.DataFrame(X_scaled)
    X.insert(loc = len(X.columns), column = 'intercept', value = 1)

    #Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

    #Training
    print("\n***TRAINING***")
    W = sgd(X_train.to_numpy(), y_train.to_numpy())
    print("***FINISHED***")
    print("\nWEIGHTS : {}".format(W))
