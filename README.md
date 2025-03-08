---

#### 📂 **Project: Neural Network for Handwritten Digit Recognition**  
A fully connected **Neural Network** built from scratch to recognize handwritten digits. 🧠✍️  
This project includes **real-time visualization** of the training process and a **drawing interface** for user input.

---

## 🌟 **Features**
✅ **Train a Neural Network from Scratch** – No deep learning frameworks, only NumPy!  
✅ **Interactive Training Visualization** – Watch the network learn with real-time updates.  
✅ **Handwritten Digit Prediction** – Draw a digit and see what the model predicts.  
✅ **Custom Neural Network Implementation** – Forward propagation, backpropagation, and training are implemented manually.  
✅ **Pygame-Based UI** – Visualize neural connections and weights dynamically.

---

## 📊 **Dataset**
This project uses the **MNIST dataset** 📄, a collection of **70,000** handwritten digits (0-9).  
- **🗂 Source**: Fetched using `fetch_openml('mnist_784')` from **scikit-learn**.  
- **🔢 Format**: Each digit is represented as a **28x28 grayscale image** (784 pixels).  
- **📌 Processing**: Normalized to [0,1] range and converted to one-hot encoded labels.  

---

## 📁 **Project Structure**
```bash
Neural-Network-Handwritten-Digit-Recognition
│── .venv/                # Virtual environment (optional)
│── data_loader.py        # Loads MNIST dataset
│── main.py               # Main script (trains model & user input interface)
│── network_visualizer.py # Visualizes training process using Pygame
│── neural_network.py     # Fully connected neural network implementation
│── utils.py              # Preprocessing functions (centering, resizing, smoothing)
│── README.md             # Project documentation
│── .gitignore            # Ignore unnecessary files
└── requirements.txt      # Required Python packages (to be generated)
```

---

## 🛠 **Installation & Setup**
### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/yourusername/Neural-Network-Handwritten-Digit-Recognition.git
cd Neural-Network-Handwritten-Digit-Recognition
```

### 2️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

#### 🏗 **Required Libraries**
- `numpy` 🧮 (Matrix operations)
- `pygame` 🎮 (GUI & visualization)
- `matplotlib` 📊 (Graphs & plots)
- `scipy` ⚙️ (Image processing)
- `scikit-learn` 🔍 (Dataset handling)
- `scikit-image` 🖼 (Resizing & preprocessing)

### 3️⃣ **Run the Training Script**
```bash
python main.py
```
🚀 The neural network will start training, and the **visualization will be displayed**.

---

## 🎨 **Drawing & Prediction**
1. 🖊 **Draw a digit** in the Pygame window.  
2. 🧠 The network processes the drawing and predicts the digit.  
3. 🎯 The predicted number is displayed in the console.

---

## 🏗 **How It Works (Neural Network Structure)**
The network follows a **fully connected architecture**:
```
Input Layer (784) → Hidden Layer 1 (512 neurons, ReLU) → Hidden Layer 2 (256 neurons, ReLU) → Output Layer (10, Softmax)
```
**🟢 Forward Propagation:**  
- Input → Weighted Sum → Activation (ReLU) → Output  
- Last layer uses **Softmax** for probability distribution.

**🔄 Backpropagation:**  
- Calculates error using **Cross Entropy Loss**.  
- Updates weights using **Gradient Descent**.  

---

## 📈 **Visualization Components**
- **Neurons**: White circles  
- **Connections**: Green (positive weights), Red (negative weights)  
- **Training Accuracy Graph**: Updates live on the right side  

---

## 📝 **Possible Enhancements**
✅ **Switch to CNNs** – Improve accuracy using convolutional layers.  
✅ **Batch Normalization** – Speed up convergence.  
✅ **Clear Button for Drawing UI** – Improve user experience.  

---

## 👨‍💻 **Author**
📌 **Lahiru Prasad**  
💬 **Contact:** 

---
