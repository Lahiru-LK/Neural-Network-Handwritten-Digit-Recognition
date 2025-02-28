import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        self.weights_input_hidden1 = np.random.randn(input_size, hidden_size_1) * 0.01
        self.weights_hidden1_hidden2 = np.random.randn(hidden_size_1, hidden_size_2) * 0.01
        self.weights_hidden2_output = np.random.randn(hidden_size_2, output_size) * 0.01
        self.bias_hidden1 = np.zeros((1, hidden_size_1))
        self.bias_hidden2 = np.zeros((1, hidden_size_2))
        self.bias_output = np.zeros((1, output_size))

    def forward(self, x):
        self.hidden1_input = np.dot(x, self.weights_input_hidden1) + self.bias_hidden1
        self.hidden1_output = np.maximum(0, self.hidden1_input)  # ReLU activation

        self.hidden2_input = np.dot(self.hidden1_output, self.weights_hidden1_hidden2) + self.bias_hidden2
        self.hidden2_output = np.maximum(0, self.hidden2_input)  # ReLU activation

        self.output_input = np.dot(self.hidden2_output, self.weights_hidden2_output) + self.bias_output
        self.output = self.softmax(self.output_input)  # Softmax activation
        return self.output

    def backward(self, x, y, output, learning_rate):
        output_error = output - y
        output_delta = output_error

        hidden2_error = np.dot(output_delta, self.weights_hidden2_output.T)
        hidden2_delta = hidden2_error * (self.hidden2_input > 0)  # ReLU derivative

        hidden1_error = np.dot(hidden2_delta, self.weights_hidden1_hidden2.T)
        hidden1_delta = hidden1_error * (self.hidden1_input > 0)  # ReLU derivative

        self.weights_hidden2_output -= learning_rate * np.dot(self.hidden2_output.T, output_delta)
        self.bias_output -= learning_rate * np.sum(output_delta, axis=0, keepdims=True)

        self.weights_hidden1_hidden2 -= learning_rate * np.dot(self.hidden1_output.T, hidden2_delta)
        self.bias_hidden2 -= learning_rate * np.sum(hidden2_delta, axis=0, keepdims=True)

        self.weights_input_hidden1 -= learning_rate * np.dot(x.T, hidden1_delta)
        self.bias_hidden1 -= learning_rate * np.sum(hidden1_delta, axis=0, keepdims=True)

    @staticmethod
    def softmax(x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    @staticmethod
    def cross_entropy_loss(y, output):
        n_samples = y.shape[0]
        log_likelihood = -np.log(output[range(n_samples), np.argmax(y, axis=1)])
        return np.sum(log_likelihood) / n_samples


