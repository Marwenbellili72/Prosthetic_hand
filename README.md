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

## 🧱 Project Architecture

### 🎨 Animation with Blender

We created a realistic 3D hand model in **Blender** by adding **armatures** (bones) and creating movement animations corresponding to different hand gestures. These animations are exported and controlled dynamically.

### 🌍 Real-time Visualization with Three.js

Using **Three.js**, we render the 3D hand model in a web browser. When a prediction is made, the corresponding animation is triggered for immediate visual feedback.

### ⚙️ Backend with FastAPI

- Loads the trained `XGBoost` model (`emg_xgboost_model.json`)
- Uses `StandardScaler` (`emg_scaler.pkl`) to preprocess incoming EMG signals
- Predicts hand gestures from EMG features
- Returns the predicted gesture to the frontend for animation

### 🐳 Dockerization

The entire stack is **Dockerized** for easy installation and deployment. This ensures consistency across environments and simplifies hosting.

### ☁️ Deployment on Hugging Face Spaces

The project is **deployed to Hugging Face Spaces**, allowing anyone to try it out online without needing to set up anything locally.

---

## 🖼️ Live Preview

<p align="center">
  <img src="static/preview.gif" width="500px" alt="3D Visualization Preview" />
</p>

> 🔗 Try the online demo on [Hugging Face Spaces](https://huggingface.co/spaces/Marwenbellili72/Prosthetic_hand) *(if public)*

---

## 📦 Installation (Local)

### 1. Clone the Repository

```bash
git clone https://github.com/Marwenbellili72/Prosthetic_hand.git
cd Prosthetic_hand
````

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

---

## 👨‍🔬 Author

* **Marwen Bellili** – [@Marwenbellili72](https://github.com/Marwenbellili72)

---

