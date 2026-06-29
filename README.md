# 🧬 CareerDNA — Neural Career Intelligence Platform

CareerDNA is an AI-powered web application that analyzes user skills and provides career insights using machine learning models and data-driven analysis.

The platform integrates multiple career intelligence tools into a single system built with **TensorFlow, Flask, and modern web technologies**.


# 🚀 Features

CareerDNA provides five AI-driven tools for career analysis.

| Feature | Description | Method |
|--------|-------------|--------|
| 🧬 Career Path Predictor | Predicts and ranks suitable career paths based on user skills | TensorFlow Artificial Neural Network (Softmax classifier) |
| 📉 Skill Decay Detector | Evaluates whether skills are rising, stable, or declining in demand | Velocity scoring + normalized scaling |
| 🎯 Interview IQ Scorer | Measures readiness for a specific job description | TF-IDF + Cosine Similarity |
| 📄 ATS Resume Checker | Performs ATS-style keyword analysis | Weighted keyword scoring |
| 🗺️ Learning Roadmap | Generates a personalized skill learning plan | Skill gap detection using similarity metrics |

The system also recommends **learning resources and missing skills** to improve career alignment.

---

# 🧠 Machine Learning Architecture

The Career Path Predictor uses a **multi-class Artificial Neural Network (ANN)**.

Architecture:

```
Input Layer: 30 skill features

Dense(128) + BatchNorm + ReLU + Dropout(0.3)
Dense(64)  + BatchNorm + ReLU + Dropout(0.2)
Dense(32)  + ReLU

Output Layer
Dense(14) + Softmax
```

Training configuration:

- Optimizer: **Adam**
- Loss: **Sparse Categorical Cross-Entropy**
- Feature Scaling: **MinMaxScaler**
- Regularization: **Dropout + L2**
- Callbacks:
  - EarlyStopping
  - ReduceLROnPlateau

The model predicts probabilities across **14 career categories**.

---

# 🧠 AI Concepts Implemented

| Concept | Implementation |
|--------|---------------|
| Multi-Class Neural Network | TensorFlow / Keras |
| Softmax Classification | Output layer |
| ReLU Activation | Hidden layers |
| Batch Normalization | Training stabilization |
| Dropout Regularization | Overfitting reduction |
| Adam Optimizer | Gradient optimization |
| Cross-Entropy Loss | Multi-class classification |
| MinMax Feature Scaling | Input normalization |
| Cosine Similarity | Skill matching |
| TF-IDF | Job description analysis |
| Early Stopping | Training optimization |
| Backpropagation | Neural network training |

---

# 📁 Project Structure

```
careerdna/
│
├── app.py
├── train_model.py
├── requirements.txt
├── runtime.txt
├── README.md
├── DOCUMENTATION.md
│
├── templates/
│   └── index.html
│
└── model/
    ├── career_model/
    │   ├── saved_model.pb
    │   ├── keras_metadata.pb
    │   ├── fingerprint.pb
    │   └── variables/
    │        ├── variables.data-00000-of-00001
    │        └── variables.index
    │
    ├── scaler.pkl
    ├── metadata.json
    └── training_history.json
```

The neural network is stored using the **TensorFlow SavedModel format**, ensuring compatibility across different environments.

---

# ⚙️ Technology Stack

## Backend
- Python
- Flask
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Pandas

## Frontend
- HTML
- CSS
- JavaScript

## Deployment
- Render Cloud Platform

---

# 💻 Running the Project Locally

### 1. Clone the repository

```
git clone https://github.com/tharsna1208/careerdna.git
```

### 2. Navigate into the project

```
cd careerdna
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Start the Flask application

```
python app.py
```

### 5. Open in browser

```
http://localhost:5000
```

---

# 🔌 API Endpoints

| Method | Endpoint | Description |
|------|---------|-------------|
| GET | / | Web application |
| GET | /api/status | System and model status |
| POST | /api/career | Career prediction |
| POST | /api/decay | Skill trend analysis |
| POST | /api/interview | Interview readiness scoring |
| POST | /api/ats | ATS resume analysis |
| POST | /api/roadmap | Personalized learning roadmap |

---

# 📦 Dependencies

Core dependencies:

- Flask
- TensorFlow
- NumPy
- Pandas
- Scikit-learn

Full dependency list available in:

- [requirements.txt](requirements.txt)
- [runtime.txt](runtime.txt)

---

# 📚 Documentation

Detailed technical explanations are available in:

- [DOCUMENTATION.md](DOCUMENTATION.md)
