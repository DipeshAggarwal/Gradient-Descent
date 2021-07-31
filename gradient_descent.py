from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

def sigmoid_activation(x):
    return 1.0 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

def predict(X, W):
    preds = sigmoid_activation(X.dot(W))
    
    preds[preds <= 0.5] = 0
    preds[preds > 0] = 1
    
    return preds

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100, help="Numbers of Epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01, help="Learning rate")
args = vars(ap.parse_args())

# Generate a 2-class classification problem with 1000 data points
X, y = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))

# Add the bias vector
X = np.c_[X, np.ones((X.shape[0]))]

trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.5, random_state=42)

print("[INFO] Training...")
W = np.random.randn(X.shape[1], 1)
losses = []

for epoch in np.arange(0, args["epochs"]):
    # Generate predictions on the dataset
    preds = sigmoid_activation(trainX.dot(W))
    
    # Determine the error, ie, the difference between preds and ground truth
    error = preds - trainY
    loss = np.sum(error ** 2)
    losses.append(loss)

    # Update the gradient descent
    d = error * sigmoid_deriv(preds)
    gradient = trainX.T.dot(d)
    
    # Nudge the weight matrix in the negative direction of the gradient
    W += -args["alpha"] * gradient
    
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1), loss))
        
print("[INFO] Evaluating...")
preds = predict(testX, W)
print(classification_report(testY, preds))

plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY[:, 0], s=30)

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
