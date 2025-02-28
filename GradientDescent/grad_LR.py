# Gradient descent for Linear Regressin
# yhat = wx + b
# total_loss = sum((yhat-y)**2) / N

import numpy as np

# Initialize parameters
x = np.random.rand(10, 1)
# y here is the answer, our goal is using ML to find out the w is 2 and b is np.random.rand()
y = 2*x + 0.168
w = np.zeros((1, 1))
b = 0.0

# Hyperparameters
epochs = 400
learning_rate = 1e-1

# Create gradient descent function
def descend(x, y, w, b, learning_rate):
    
    N = x.shape[0]
    
    yhat = np.dot(x, w) + b
    error = yhat - y
    
    dldw = 2 * np.dot(x.T, error) / N
    dldb = 2 * np.sum(error) / N
        
    w = w - learning_rate * dldw
    b = b - learning_rate * dldb
    
    return w, b

# Iteratively make updates
for epoch in range(epochs):
    w, b = descend(x, y, w, b , learning_rate)
    yhat = np.dot(x, w) + b
    loss = np.sum((yhat-y)**2, axis=0) / x.shape[0]
    print(f'epoch: {epoch} loss: {loss}; w: {w}, b: {b}')
