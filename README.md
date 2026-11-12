# 🚦 Traffic Sign Recognition using CNN

A deep learning project that classifies German traffic signs into 43 categories using a Convolutional Neural Network (CNN) trained on the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset. The project includes a full training pipeline built in Google Colab, along with **Streamlit** and **Gradio** interfaces for real-time image prediction.

---

## 📌 Features

- CNN model trained from scratch on 43 traffic sign classes
- Data preprocessing pipeline (resizing, normalization, stratified train/val split)
- Achieves high accuracy on the official GTSRB test set
- Interactive **Streamlit** web app for local predictions
- **Gradio** interface for quick demos in Google Colab
- Top-3 prediction output with confidence scores

---

## 🗂️ Dataset

- **Name:** GTSRB - German Traffic Sign Recognition Benchmark
- **Classes:** 43
- **Source:** [Kaggle - GTSRB Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- Images vary in size, lighting, and angle, making the model robust to real-world conditions.

---

## 🧠 Model Architecture

| Layer | Details |
|---|---|
| Conv2D | 32 filters, 5x5 kernel, ReLU |
| Conv2D | 32 filters, 5x5 kernel, ReLU |
| MaxPooling2D | 2x2 |
| Dropout | 0.25 |
| Conv2D | 64 filters, 3x3 kernel, ReLU |
| Conv2D | 64 filters, 3x3 kernel, ReLU |
| MaxPooling2D | 2x2 |
| Dropout | 0.25 |
| Flatten | — |
| Dense | 256 units, ReLU |
| Dropout | 0.5 |
| Dense (Output) | 43 units, Softmax |

**Loss:** Categorical Crossentropy · **Optimizer:** Adam · **Epochs:** 20 · **Batch Size:** 64

---

## 📁 Project Structure

```
traffic-sign-recognition/
│
├── traffic_sign_model.h5        # Trained CNN model
├── app.py                       # Streamlit app for local inference
├── requirements.txt             # Python dependencies
├── notebook/
│   └── Traffic_Sign_Recognition.ipynb   # Colab training notebook
├── docs/
│   └── Project_Documentation.md # Full project documentation
└── README.md
```

---

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/traffic-sign-recognition.git
cd traffic-sign-recognition
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`. Upload a traffic sign image to get a prediction with confidence scores.

---

## 🏋️ Training the Model (Optional)

The full training pipeline is available in the notebook under `notebook/Traffic_Sign_Recognition.ipynb`, designed to run on **Google Colab** with GPU support. It covers:

1. Downloading the GTSRB dataset via Kaggle API
2. Preprocessing images (resize to 32x32, normalize)
3. Building and training the CNN
4. Evaluating on the test set
5. Saving the trained model (`.h5`)

---

## 📊 Results

- **Test Accuracy:** ~95–98% on the official GTSRB test set
- Evaluated using accuracy, precision, recall, and F1-score (via `classification_report`) to account for class imbalance across the 43 sign categories

---

## 🚀 Future Improvements

- Data augmentation for underrepresented classes
- Transfer learning with pretrained models (MobileNet, ResNet)
- Real-time sign detection using YOLO + this classifier
- TensorFlow Lite conversion for edge/embedded deployment
- Grad-CAM visualizations for model explainability

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- Scikit-learn
- Streamlit
- Gradio
- Google Colab

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙋 Author

Feel free to reach out if you have questions or suggestions for improving this project!

## Student Submission

Name: Akshita Goel
UID: 25MDA10107
Course: MBA Data Science and AI

## Project Overview

This project focuses on traffic sign classification
using a Convolutional Neural Network (CNN).

Original source:
https://github.com/CMSMalshan/Traffic-Sign-Recognition
