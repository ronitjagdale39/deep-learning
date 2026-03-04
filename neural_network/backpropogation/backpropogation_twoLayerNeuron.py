import numpy as np
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)
def initialize_parameters(input_size, h1_size, h2_size, output_size):
    np.random.seed(42)

    W1 = np.random.randn(h1_size, input_size) * 0.01
    b1 = np.zeros((h1_size, 1))

    W2 = np.random.randn(h2_size, h1_size) * 0.01
    b2 = np.zeros((h2_size, 1))

    W3 = np.random.randn(output_size, h2_size) * 0.01
    b3 = np.zeros((output_size, 1))

    return W1, b1, W2, b2, W3, b3
def forward(X, W1, b1, W2, b2, W3, b3):
    
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)

    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, A1, Z2, A2, Z3, A3)
    return A3, cache
def compute_loss(Y, A3):
    m = Y.shape[1]
    loss = - (1/m) * np.sum(Y*np.log(A3 + 1e-8) + (1-Y)*np.log(1-A3 + 1e-8))
    return loss
def backward(X, Y, cache, W2, W3):
    
    m = X.shape[1]
    Z1, A1, Z2, A2, Z3, A3 = cache


    dZ3 = A3 - Y
    dW3 = (1/m) * np.dot(dZ3, A2.T)
    db3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True)


    dA2 = np.dot(W3.T, dZ3)
    dZ2 = dA2 * relu_derivative(Z2)
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)


    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * sigmoid_derivative(Z1)
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3
def update(W1, b1, W2, b2, W3, b3,
           dW1, db1, dW2, db2, dW3, db3,
           lr):

    W1 -= lr * dW1
    b1 -= lr * db1

    W2 -= lr * dW2
    b2 -= lr * db2

    W3 -= lr * dW3
    b3 -= lr * db3

    return W1, b1, W2, b2, W3, b3

def train(X, Y, epochs=1000, lr=0.1):
    
    input_size = X.shape[0]
    W1, b1, W2, b2, W3, b3 = initialize_parameters(input_size, 8, 6, 1)

    for i in range(epochs):
        
        A3, cache = forward(X, W1, b1, W2, b2, W3, b3)
        loss = compute_loss(Y, A3)

        grads = backward(X, Y, cache, W2, W3)

        W1, b1, W2, b2, W3, b3 = update(
            W1, b1, W2, b2, W3, b3,
            *grads,
            lr
        )

        if i % 100 == 0:
            print(f"Epoch {i}, Loss: {loss:.4f}")

    return W1, b1, W2, b2, W3, b3