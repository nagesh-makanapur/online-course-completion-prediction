# 🎓 Online Course Completion Prediction (FastAPI + ML)

This project predicts whether a learner will complete an online course based on demographic and engagement features.  
It includes both **training pipeline** and an **inference API** (FastAPI) for serving predictions.

---

## 📂 Project Structure
````
Online-Course-Completion-ML/
│── app/ # FastAPI application
│ ├── main.py # FastAPI entrypoint
│ ├── inference.py # Inference class (loads model & preprocessing)
│ ├── init.py
│
│── data/ # Dataset
│ └── online_course_completion.csv
│
│── models/ # Saved ML model
│ └── model.pkl
│
│── notebooks/ # Jupyter notebooks for EDA & training
│ └── training.ipynb
│
│── requirements.txt # Python dependencies
│── README.md # Project documentation


---

## ⚙️ Installation

Clone the repo:

```bash
git clone https://github.com/nagesh-makanapur /Online-Course-Completion-ML.git
cd Online-Course-Completion-ML
````
Install dependencies:
```
pip install -r requirements.txt
```
📊 Dataset

The dataset used is:

File: data/online_course_completion.csv

Features: learner demographics + engagement statistics

Target: completed (1 = completed, 0 = not completed)

🚀 Training

To train the model, use the Jupyter Notebook in notebooks/training.ipynb.
It handles preprocessing (scaling, encoding) and saves the model as models/model.pkl.

🤖 Inference API (FastAPI)

The inference class loads the trained model & preprocessing pipeline, then exposes a predict method.
It is wrapped with FastAPI for serving predictions.

Run FastAPI server:
``````
python -m uvicorn app.main:app --reload
```````
🛠 Requirements
All dependencies are listed in requirements.txt:




