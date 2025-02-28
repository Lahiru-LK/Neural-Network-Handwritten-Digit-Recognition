from sklearn.datasets import fetch_openml
import numpy as np

def load_data():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(int)
    X /= 255.0
    y_onehot = np.eye(10)[y]

    print(f"✅ Data Loaded: {X.shape}, Labels: {y_onehot.shape}")

    split_index = 60000
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y_onehot[:split_index], y_onehot[split_index:]

    return X_train, X_test, y_train, y_test
