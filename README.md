
# AquaAlert Model

A machine learning-based solution for anomaly detection in industrial parameters such as air temperature, flow speed, and pressure. This project leverages LSTM models to predict and classify system anomalies.

---

## <a id='someLabel' href="https://colab.research.google.com/drive/1fHMh8DUUzaxumXCuQ25nDwEL1K5uik5N?usp=sharing">Colab Link</a> 

## Features

- **Inputs**: 
  - Air Temperature [K]
  - Flow Speed
  - Pressure [N]

- **Target**: Binary classification of anomalies (`0` for normal, `1` for anomaly).

- **Model**:
  - Sequential LSTM-based neural network for time-series data.
  - Custom class balancing with `compute_class_weight`.

---

## Data

### Dataset Overview

| Column                | Description                         |
|-----------------------|-------------------------------------|
| `Air temperature [K]` | Air temperature in Kelvin.          |
| `flow speed`          | Flow speed of the system.           |
| `pressure [N]`        | Pressure measured in Newtons.       |
| `Target`              | Binary classification target (0/1). |
| `DateTime`            | Timestamp for each observation.     |

### Exploratory Data Analysis (EDA)

Key insights from the dataset:
1. **Distributions**:
   - Visualized feature distributions to understand data spread.
2. **Correlation Heatmap**:
   - Identified relationships between numerical features and target variable.

---

## Model Architecture

The LSTM model is designed as follows:

1. **LSTM Layer 1**: 64 units, ReLU activation.
2. **Dropout Layer**: 20% dropout for regularization.
3. **LSTM Layer 2**: 32 units.
4. **Dense Layer**: 16 units with ReLU activation.
5. **Output Layer**: Sigmoid activation for binary classification.

---

## Installation

### Requirements
- Python 3.x
- TensorFlow
- Scikit-learn
- Matplotlib
- NumPy
- Pandas
- Seaborn
- Joblib

Install dependencies:
```bash
pip install tensorflow scikit-learn matplotlib numpy pandas seaborn joblib
```

---

## Usage

1. **Data Preprocessing**:
   - Scaled features using `MinMaxScaler`.
   - Reshaped data for LSTM input format.

2. **Model Training**:
   - Used balanced class weights to address imbalanced classes.
   - Trained over 30 epochs with early stopping and validation monitoring.

3. **Prediction**:
   - Predicts anomalies based on new input data.

### Predicting on New Data

```python
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load the scaler and model
scaler = joblib.load('scaler.pkl')
model = load_model('aquaAlert.pkl')

# New input data
new_data = np.array([[301.8, 1379, 52.3]])

# Preprocess
scaled_data = scaler.transform(new_data)
reshaped_data = scaled_data.reshape((scaled_data.shape[0], 1, scaled_data.shape[1]))

# Predict
predictions = model.predict(reshaped_data)
predicted_class = (predictions > 0.5).astype(int)

print("Predicted probability:", predictions)
print("Predicted class:", predicted_class)
```

---

## Visualizations

### Feature Distributions

- **Air Temperature [K]**
- **Flow Speed**
- **Pressure [N]**

### Correlation Heatmap

Highlights correlations between features and the target variable.

---

## Results

- Achieved a balanced classification with class weights.
- **Classification Metrics**:
  - Precision, Recall, F1-score for anomaly detection.

---

## Files in Repository

- `last.csv`: Dataset used for training and testing.
- `aquaAlert.pkl`: Trained LSTM model.
- `scaler.pkl`: MinMaxScaler for feature scaling.
- `Model.ipynb`: Jupyter notebook for the complete pipeline.

---

## Visualizations of Dataset 
![output](https://github.com/user-attachments/assets/f607527c-f97b-4e6f-ace3-e8c2b1a46231)
![output(1)](https://github.com/user-attachments/assets/d9850aad-7239-463c-bffb-157d09d2f3d4)
![output(2)](https://github.com/user-attachments/assets/ede4ab61-48f3-46a2-986a-22b5c3e6c288)
![output(3)](https://github.com/user-attachments/assets/9e53665b-cab7-48b6-9b0b-dd3e3a92d026)

