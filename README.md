# 🧠 LSTM Time-Series Prediction

A simple Deep Learning project that uses **LSTM (Long Short-Term Memory)** networks to predict the next value in a time-series sequence.

This project is designed to understand how LSTM works, why it is useful for sequential data, and how to build a basic LSTM model using Python and TensorFlow/Keras.

---

## 📌 Project Objective

The objective of this project is to:

* Understand the concept of **LSTM**
* Prepare sequential/time-series data
* Create input sequences using previous values
* Scale data using `MinMaxScaler`
* Build an LSTM neural network
* Train the model using TensorFlow/Keras
* Predict future values
* Compare actual vs predicted values
* Understand LSTM concepts for interviews

---

## 🔄 Project Workflow

```text
Time-Series Data
       ↓
Data Scaling
       ↓
Create Sequences
       ↓
Reshape Data
       ↓
LSTM Layer
       ↓
LSTM Layer
       ↓
Dense Layer
       ↓
Prediction
       ↓
Actual vs Predicted
```

---

## 📂 Project Structure

```text
LSTM-Time-Series-Prediction/
│
├── lstm_prediction.py
├── README.md
└── requirements.txt
```

---

## 🛠️ Technologies Used

* Python
* NumPy
* Matplotlib
* Scikit-learn
* TensorFlow
* Keras
* LSTM
* Deep Learning

---

# 🧠 What is LSTM?

**LSTM stands for Long Short-Term Memory.**

It is a type of **Recurrent Neural Network (RNN)** designed to work with sequential data.

Traditional neural networks generally treat observations independently, while LSTM can use information from previous time steps to help make predictions.

For example:

```text
Day 1 → Day 2 → Day 3 → Day 4 → Day 5
                         ↓
                    Prediction
```

LSTM is useful when the order of the data matters.

---

# 🔥 Why LSTM?

LSTM was designed to address the **vanishing gradient problem** that can occur in traditional RNNs.

It can learn both:

* Short-term dependencies
* Long-term dependencies

Common applications include:

* Stock-price/time-series forecasting
* Demand forecasting
* Weather prediction
* Speech recognition
* Text generation
* Sentiment analysis
* Predictive maintenance
* Sequence classification

---

# 🔐 LSTM Architecture

An LSTM cell mainly contains three gates:

```text
             ┌──────────────┐
Previous ───→│ Forget Gate  │
             └──────────────┘
                    ↓
             ┌──────────────┐
Current ────→│ Input Gate   │
             └──────────────┘
                    ↓
             ┌──────────────┐
             │ Cell State   │
             └──────────────┘
                    ↓
             ┌──────────────┐
             │ Output Gate  │
             └──────────────┘
                    ↓
                 Output
```

### 1. Forget Gate

Decides which information should be removed from the previous cell state.

```text
What information is no longer important?
```

### 2. Input Gate

Decides which new information should be added to the cell state.

```text
What new information should we remember?
```

### 3. Output Gate

Decides what information should be passed to the next layer/time step.

```text
What information should we output?
```

---

# 📊 Dataset

For this beginner project, a synthetic sine-wave time series is generated using NumPy.

```python
data = np.sin(np.arange(0, 100, 0.1))
```

The advantage of using a generated dataset is that we can focus on understanding the LSTM architecture without spending time on data cleaning.

---

# 🔢 Creating Sequences

LSTM requires sequential input.

In this project:

```python
sequence_length = 50
```

The previous **50 values** are used to predict the next value.

Example:

```text
Input:

[1, 2, 3, 4, 5, ... 50]

Target:

51
```

Then:

```text
[2, 3, 4, 5, ... 51] → 52
```

This process creates many training sequences.

---

# 🔄 Data Scaling

The data is scaled between 0 and 1 using:

```python
MinMaxScaler(feature_range=(0, 1))
```

Why?

Neural networks generally perform better when numerical inputs are on a similar scale.

```text
Original Data
     ↓
MinMaxScaler
     ↓
0 to 1
     ↓
LSTM
```

---

# 📐 LSTM Input Shape

One of the most important concepts in LSTM is the input shape.

The input has three dimensions:

```text
(samples, timesteps, features)
```

In this project:

```python
X.shape
```

is approximately:

```text
(samples, 50, 1)
```

Meaning:

```text
Samples   → Number of training sequences

Timesteps → 50 previous values

Features  → 1 value at each timestep
```

---

# 🏗️ Model Architecture

The model contains two LSTM layers followed by a Dense layer.

```python
model = Sequential()

model.add(
    LSTM(
        50,
        return_sequences=True,
        input_shape=(X.shape[1], 1)
    )
)

model.add(
    LSTM(50)
)

model.add(
    Dense(1)
)
```

Architecture:

```text
Input
  ↓
LSTM(50)
  ↓
LSTM(50)
  ↓
Dense(1)
  ↓
Prediction
```

---

# 🔁 Why `return_sequences=True`?

The first LSTM layer is followed by another LSTM layer.

Therefore, the first LSTM needs to return the complete sequence.

```python
return_sequences=True
```

Without it, the first LSTM would normally return only its final output, which would not provide the expected sequence input to the next LSTM layer.

---

# ⚙️ Model Compilation

```python
model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)
```

### Adam

Adam is an optimization algorithm used to update the neural-network weights during training.

### Mean Squared Error

MSE measures the difference between actual and predicted values.

Conceptually:

```text
Actual Value
     ↓
Difference
     ↓
Squared Difference
     ↓
Average
     ↓
MSE
```

Lower loss generally means the model's predictions are closer to the target values.

---

# 🚀 Model Training

```python
model.fit(
    X,
    y,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)
```

### Epoch

One complete pass through the training data.

### Batch Size

Number of samples processed before the model updates its weights.

### Validation Split

20% of the available training data is used for validation.

---

# 📈 Prediction

After training:

```python
predictions = model.predict(X)
```

The predictions are then converted back to the original scale:

```python
predictions = scaler.inverse_transform(predictions)
```

The actual and predicted values can then be plotted.

---

# 📊 Expected Result

The project generates a graph similar to:

```text
Value
  |
  |       Actual
  |      /\/\      /\/\
  |     /    \    /    \
  |    /      \__/      \
  |
  |       Predicted
  |      /\/\      /\/\
  |     /    \    /    \
  |____/______\__/______\________ Time
```

The objective is for the predicted curve to follow the underlying time-series pattern.

---

# 🆚 RNN vs LSTM

| Feature                | RNN              | LSTM            |
| ---------------------- | ---------------- | --------------- |
| Sequential data        | ✅                | ✅               |
| Memory                 | Shorter          | Longer          |
| Vanishing gradient     | More susceptible | Better handling |
| Architecture           | Simpler          | More complex    |
| Gates                  | ❌                | ✅               |
| Long-term dependencies | Difficult        | Better          |
| Computation            | Lower            | Higher          |

---

# 🆚 LSTM vs GRU

| Feature                | LSTM          | GRU                    |
| ---------------------- | ------------- | ---------------------- |
| Gates                  | 3 main gates  | 2 main gates           |
| Cell state             | Separate      | No separate cell state |
| Complexity             | Higher        | Lower                  |
| Training               | Can be slower | Often faster           |
| Long-term dependencies | Excellent     | Excellent              |
| Parameters             | More          | Fewer                  |

---

# 🎯 Real-World Applications

LSTM can be used in many real-world problems:

### Finance

```text
Historical Data
      ↓
LSTM
      ↓
Future Trend Prediction
```

### Demand Forecasting

```text
Previous Demand
      ↓
LSTM
      ↓
Future Demand
```

### Weather

```text
Historical Temperature
      ↓
LSTM
      ↓
Future Temperature
```

### NLP

```text
Previous Words
      ↓
LSTM
      ↓
Next Word
```

---

# ❓ Interview Questions

### Beginner

**1. What is LSTM?**

LSTM is a type of recurrent neural network designed to learn patterns and dependencies in sequential data.

**2. Why is LSTM better than traditional RNN for long sequences?**

LSTM uses a cell state and gates to control the flow of information, helping it retain important information for longer periods.

**3. What are the gates in LSTM?**

The main gates are:

* Forget Gate
* Input Gate
* Output Gate

**4. What is the cell state?**

The cell state acts as a memory path that carries important information across time steps.

**5. Why do we scale the data?**

Scaling helps keep numerical values within a suitable range and can improve neural-network training.

---

### Intermediate

**6. What does `(samples, timesteps, features)` mean?**

```text
samples  → Number of sequences
timesteps → Number of previous observations
features → Number of variables at each timestep
```

**7. Why use `return_sequences=True`?**

When another recurrent layer follows, the previous LSTM needs to return the sequence rather than only the final output.

**8. What is the purpose of `Dense(1)`?**

It produces a single numerical prediction.

**9. What is overfitting?**

Overfitting occurs when the model learns the training data too closely and performs poorly on unseen data.

**10. How can we reduce LSTM overfitting?**

Possible approaches include:

* Dropout
* Early stopping
* More training data
* Reducing model complexity
* Regularization

---

# 💡 Important LSTM Concepts

```text
RNN
 ↓
Problem: Vanishing Gradient
 ↓
LSTM
 ↓
Cell State + Gates
 ↓
Better Long-Term Memory
```

The most important idea to remember:

> **LSTM controls what to remember, what to forget, and what to output.**

---

# ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Python file

```bash
python lstm_prediction.py
```

---

# 📚 Key Learning Outcomes

After completing this project, you should understand:

* What LSTM is
* Why LSTM is used
* RNN vs LSTM
* LSTM vs GRU
* LSTM gates
* Cell state
* Time steps
* Sequence creation
* Data scaling
* LSTM input shape
* `return_sequences`
* Model training
* Loss function
* Prediction
* Overfitting

---

# 🚀 Future Improvements

This beginner project can be extended into a more realistic forecasting project.

### Version 2

Use a real-world dataset:

```text
Stock Price
Electricity Consumption
Temperature
Airline Passengers
Sales
```

### Version 3

Compare:

```text
Linear Regression
      ↓
Random Forest
      ↓
XGBoost
      ↓
LSTM
```

### Version 4

Build a Streamlit application:

```text
CSV Upload
    ↓
Data Visualization
    ↓
LSTM Model
    ↓
Prediction
    ↓
Interactive Chart
```

---

# 👨‍💻 Author

**Subrata Mondal**

Data Analytics | Power BI | Python | Machine Learning | Deep Learning | AI

---

⭐ If you find this project useful, consider giving the repository a star.
