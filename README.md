# 🦾 Prosthetic Hand – Intelligent Myoelectric Control System

An advanced myoelectric prosthetic hand project that leverages machine learning (XGBoost), EMG signal processing, 3D real-time visualization, and web deployment.

---

## 🧩 Problem Statement

Traditional myoelectric prosthetic hands rely heavily on **linear control systems**, which limit the ability to perform complex movements. These conventional approaches are often constrained by:

- Basic open/close gestures only  
- Low robustness to electrode displacement or muscle fatigue  
- No visual feedback for calibration or validation

These limitations hinder the usability and intuitiveness of prosthetic hands for users who require more versatile, natural control.

---

## 📚 State of the Art

Modern approaches in prosthetics employ **machine learning** algorithms to interpret EMG (electromyographic) signals more effectively:

- Support Vector Machines (SVM)  
- Neural Networks  
- Random Forests  
- Decision Trees

These models offer better gesture classification but require a full-stack implementation, including signal processing, model training, animation handling, and deployment.

---

## ✅ Our Solution

We developed an intelligent prosthetic hand system featuring:

- 🧠 **XGBoost** model trained on EMG signals to classify hand gestures  
- ⚡ **FastAPI** to serve the prediction model through an API  
- 🖐️ **Blender** to create and animate a realistic 3D hand model using bone rigging (armatures)  
- 🌐 **Three.js** to render and animate the hand in the browser based on predictions  
- 🐳 **Docker** to containerize the full project for reproducibility  
- ☁️ **Hugging Face Spaces** to deploy the system publicly for demo and testing  

---

## ✊ Recognized Hand Movements

- Rest  
- Index finger flexion  
- Thumb abduction  
- Thumb adduction  
- Thumb flexion  
- Middle finger flexion  
- Index finger extension  
- Middle finger extension  
- Ring finger flexion  
- Ring finger extension  
- Little finger flexion  
- Little finger extension  
- Thumb extension  

---

## 📊 Dataset

The model was trained using the [**NinaPro Database 1 (DB1)**](https://ninapro.hevs.ch/instructions/DB1.html), which contains high-quality EMG signals from multiple subjects performing various hand gestures.

### 🔍 Selected Data

We only used **Exercise 1** from NinaPro DB1, which focuses on **individual finger movements and resting state**, aligning with our animated gesture set.

---

## 🧪 Try it Yourself (on Hugging Face)

You can **upload your own EMG data file** (preprocessed to match Exercise 1 structure) directly through the web interface on **Hugging Face Spaces**, and the system will:

1. Process the signal  
2. Predict the gesture  
3. Show the animation in real time  

> 🔗 [Hugging Face Space Demo](https://huggingface.co/spaces/Marwenbellili72/Prosthetic_hand)

---

## 📈 Model Performance

To evaluate the model's accuracy and performance, refer to the included **Jupyter Notebook** file:

```bash
notebooks/model_training_and_evaluation.ipynb
````

It contains:

* Preprocessing steps
* Feature extraction
* Model training using XGBoost
* Performance metrics (accuracy, confusion matrix, etc.)
  
The notebook also includes:

* 📊 A confusion matrix plot to analyze classification results
* 📉 Metrics such as accuracy, precision, recall, and F1-score
* 📌 Step-by-step code for reproducibility and understanding of the training process
---
🔍 To view the training code and see how the model was built and evaluated, please check the .ipynb file directly.

---

## 🧱 Project Architecture

### 🎨 Animation with Blender

We created a realistic 3D hand model in **Blender** by adding **armatures** (bones) and designing movement animations corresponding to each gesture. These animations were exported and dynamically controlled via code.

### 🌍 Real-time Visualization with Three.js

Using **Three.js**, we render the 3D hand model in a web browser. When a prediction is made, the corresponding animation is triggered for immediate visual feedback.

### ⚙️ Backend with FastAPI

* Loads the trained `XGBoost` model (`emg_xgboost_model.json`)
* Uses `StandardScaler` (`emg_scaler.pkl`) to preprocess incoming EMG signals
* Predicts hand gestures from EMG features
* Returns the predicted gesture to the frontend for animation

### 🐳 Dockerization

The full project is **Dockerized** to simplify deployment and ensure reproducibility.

### ☁️ Deployment on Hugging Face Spaces

The application is deployed on **Hugging Face Spaces**, offering a browser-accessible environment to test the prosthetic hand system without installation.

---

## 🖼️ Live Preview

<p align="center">
  <img src="static/preview.gif" width="500px" alt="3D Visualization Preview" />
</p>

---

## 📦 Installation (Local)

### 1. Clone the Repository

```bash
git clone https://github.com/Marwenbellili72/Prosthetic_hand.git
cd Prosthetic_hand
```

### 2. Run with Docker

```bash
docker build -t prosthetic-hand .
docker run -p 8000:8000 prosthetic-hand
```

### 3. Run Manually (Dev Mode)

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn main_api:app --reload
```

---

## 🗂️ Project Structure

```
Prosthetic_hand/
├── main_api.py
├── emg_xgboost_model.json
├── emg_scaler.pkl
├── notebooks/
│   └── model_training_and_evaluation.ipynb
├── static/
│   ├── models/
│   │   └── hand_model.glb
│   ├── js/
│   │   └── viewer.js
│   └── preview.gif
├── templates/
│   └── index.html
├── requirements.txt
├── Dockerfile
├── README.md
```
