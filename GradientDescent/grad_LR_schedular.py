# Gradient descent for Linear Regressin
# yhat = wx + b
# total_loss = sum((yhat-y)**2) / N

import numpy as np
np.random.seed(42) 
# Initialize parameters
x = np.random.rand(10, 1)
y = 2*x + 0.168
w = np.zeros((1, 1))
b = 0.0

# Hyperparameters
epochs = 800
init_lr = 1e-1

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

def lr_scheuler(epoch, initial_lr, scheduler_type='step', decay_rate=0.9, step_size=100):
    """
    Adjust the learning rate based on epoch number.
    
    Args:
        epoch: Current epoch number
        initial_lr: Initial learning rate
        scheduler_type: Type of scheduler ('step', 'exponential', or 'inverse')
        decay_rate: Rate at which to decay the learning rate
        step_size: Number of epochs after which to apply decay (for 'step)
    
    Return:
        Updated learning rate
    """
    
    if scheduler_type == 'step':
        return initial_lr * (decay_rate ** (epoch // step_size))
    elif scheduler_type == 'exp':
        return initial_lr * np.exp(-decay_rate * epoch)
    elif scheduler_type == 'inverse':
        return initial_lr / (1 + decay_rate * epoch)
    else:
        return initial_lr

learning_rates = []
# Iteratively make updates
for epoch in range(epochs):
    current_lr = lr_scheuler(
        epoch, init_lr, scheduler_type='step', decay_rate=0.95, step_size=50
    )
    learning_rates.append(current_lr)
    
    w, b = descend(x, y, w, b , current_lr)
    yhat = np.dot(x, w) + b
    loss = np.sum((yhat-y)**2, axis=0) / x.shape[0]
    # Print progress every 20 epochs
    if epoch % 20 == 0 or epoch == epochs - 1:
        print(f'epoch: {epoch}, loss: {loss}, w: {w[0][0]:.6f}, b: {b:.6f}, lr: {current_lr:.6f}')
