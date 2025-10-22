# 🩺 Diabetes Risk Prediction (Exploratory Data Analysis)

## 📊 Project Overview
This project explores health data to find out what factors (like BMI and age) increase the risk of diabetes.
We clean, analyze, and visualize data to get useful insights.

---

## 🧠 Objectives
1. Load and clean data.
2. Explore patterns using charts.
3. Create a simple rule to find “High Risk” patients.
4. Build a small machine learning model.

---

## 🧰 Tools Used
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## ⚙️ How It Works
### Step 1: Load Data
```python
import pandas as pd
df = pd.read_csv("diabetes_sample.csv")
df.head()

