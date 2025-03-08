# 🚀 Neural Network for Handwritten Digit Recognition 🧠✍️

This project is a **custom-built neural network** that recognizes handwritten digits from the **MNIST dataset** and provides **real-time training visualization**. Users can also **draw a digit**, and the trained model will predict the number. 🔢✨

---

## 🌟 Features
✅ **Train a Fully Connected Neural Network from Scratch** (No TensorFlow or PyTorch!)  
✅ **Interactive Training Visualization** – Observe neuron activations, weights, and accuracy in real-time 🎨📊  
✅ **Live Drawing Interface** – Draw a digit and get predictions instantly ✍️🤖  
✅ **Custom Implementation of Forward & Backpropagation** – Full control over weight updates and learning process ⚙️  
✅ **Optimized Training with Mini-Batches** – Speeds up learning and improves accuracy 📈  
✅ **Dependencies Managed via requirements.txt** – Easily install all required libraries 📦

---

## 📊 Dataset: MNIST
This project uses the **MNIST dataset**, a collection of **70,000 handwritten digits (0-9)**.

🔹 **Where is the dataset loaded from?**  
📌 The dataset is **fetched from OpenML** using `scikit-learn`:
```python
from sklearn.datasets import fetch_openml
mnist_data = fetch_openml(name="mnist_784", version=1, as_frame=False)
```

🔹 **How is the data processed?**  
✅ **Normalization:** Pixel values are scaled between **[0,1]**.  
✅ **One-hot Encoding:** Converts labels (0-9) into a **binary vector** format.  
✅ **Training & Testing Split:** The dataset is divided into **60,000 training** and **10,000 test samples**.  

---

## 📂 Project Structure
```bash
📦 Neural-Network-Handwritten-Digit-Recognition
 ┣ 📂 .venv/                 # Virtual environment (optional)
 ┣ 📜 .gitignore             # Ignore unnecessary files
 ┣ 📜 data_loader.py         # Loads and preprocesses the MNIST dataset
 ┣ 📜 main.py                # Main script: training, visualization, and prediction
 ┣ 📜 network_visualizer.py  # Visualizes neural network structure using Pygame
 ┣ 📜 neural_network.py      # Neural Network implementation (forward & backpropagation)
 ┣ 📜 utils.py               # Digit preprocessing functions (resize, normalize, center)
 ┣ 📜 requirements.txt       # Required dependencies
 ┣ 📜 README.md              # Project documentation (this file)
```

---

## 📦 Required Libraries
```bash
numpy           # Matrix operations 🧮
pygame          # GUI & visualization 🎮
matplotlib      # Graphs & plots 📊
scipy           # Image processing ⚙️
scikit-learn    # Dataset handling 🔍
scikit-image    # Resizing & preprocessing 🖼
```

To install all dependencies, run:
```bash
pip install -r requirements.txt
```

---

## 🛠 Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/Neural-Network-Handwritten-Digit-Recognition.git
cd Neural-Network-Handwritten-Digit-Recognition
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Training Script
```bash
python main.py
```
🚀 The neural network will start training, and a **visualization window** will open.

---

## 🎨 Drawing & Prediction
1. **A Pygame window will appear** where you can draw a digit. ✍️  
2. **The network processes the image** and predicts the digit. 🧠  
3. **The result is displayed in the console.** 🎯  

---

## 🏗 Neural Network Architecture
```
Input Layer (784) → Hidden Layer 1 (512 neurons, ReLU) → Hidden Layer 2 (256 neurons, ReLU) → Output Layer (10, Softmax)
```
📌 **Forward Propagation**
```python
self.hidden1_input = np.dot(x, self.weights_input_hidden1) + self.bias_hidden1
self.hidden1_output = np.maximum(0, self.hidden1_input)  # ReLU Activation
self.hidden2_input = np.dot(self.hidden1_output, self.weights_hidden1_hidden2) + self.bias_hidden2
self.hidden2_output = np.maximum(0, self.hidden2_input)  # ReLU Activation
self.output_input = np.dot(self.hidden2_output, self.weights_hidden2_output) + self.bias_output
return self.softmax(self.output_input)
```

📌 **Backpropagation**
```python
output_error = output - y
hidden2_error = np.dot(output_error, self.weights_hidden2_output.T) * (self.hidden2_input > 0)
hidden1_error = np.dot(hidden2_error, self.weights_hidden1_hidden2.T) * (self.hidden1_input > 0)
```

---

## 📊 Prediction Accuracy
During training, the model achieves **high accuracy**, and test accuracy typically reaches around **97-99%**. 🎯  
Each epoch prints the current accuracy in the console, showing improvement over time:
```plaintext
Epoch 10/20 -> Train Accuracy: 96.57%
Test Accuracy: 97.52%
Epoch 15/20 -> Train Accuracy: 99.38%
Test Accuracy: 97.92%
Epoch 20/20 -> Train Accuracy: 99.83%
Test Accuracy: 97.99%
```

---

## 🔧 Technologies Used
| Feature                 | Library        |
|-------------------------|---------------|
| **Machine Learning**    | NumPy, scikit-learn |
| **Visualization**       | Pygame        |
| **Data Processing**     | Scipy, scikit-image |
| **Mathematics**        | NumPy (Matrix operations) |
| **Neural Network**      | Fully custom Python implementation |

---


