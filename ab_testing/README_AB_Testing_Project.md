# 🧪 A/B Testing – E-Commerce Conversion Analysis

### 📘 Project Overview
This project simulates and analyzes an **A/B test** for an e-commerce website to determine whether a **new website design (Group B)** leads to a higher **conversion rate** compared to the **current design (Group A)**.

The analysis covers **data creation**, **exploratory data analysis (EDA)**, and **hypothesis testing** to make a statistically-sound business recommendation.

---

## 🎯 Objectives
1. Simulate a realistic e-commerce A/B testing dataset.  
2. Explore the dataset using pandas and visualize group performance.  
3. Use a **two-proportion Z-test** to determine if the observed difference in conversions is statistically significant.  
4. Communicate insights and recommendations for decision-making.

---

## 🧰 Tools & Libraries
| Tool | Purpose |
|------|----------|
| **Python 3** | Core programming language |
| **pandas** | Data manipulation and EDA |
| **NumPy** | Numeric computation |
| **matplotlib / seaborn** | Data visualization |
| **SciPy** | Hypothesis testing (Z-test) |
| **Jupyter Notebook / VS Code** | Interactive analysis environment |

---

## 📂 Dataset Description
File: `ab_test_results.csv`  
- **user_id:** Unique identifier for each user  
- **group:** Test group (A = control, B = variant)  
- **converted:** Binary outcome — 1 if user converted, 0 otherwise  

Dataset size: 10 000 users  
- Group A (control): 5 000 users, ~10 % conversion  
- Group B (variant): 5 000 users, ~12 % conversion  

---

## 🧠 Steps & Workflow

### 1️⃣ Data Creation
Synthetic data generated using a **binomial distribution**:
```python
group_A = np.random.binomial(1, 0.10, 5000)
group_B = np.random.binomial(1, 0.12, 5000)
```
Combined into a single DataFrame and saved as `ab_test_results.csv`.

---

### 2️⃣ Exploratory Data Analysis (EDA)
- Verified data structure and types.  
- Checked missing values (none).  
- Calculated group-wise conversion rates.  
- Visualized differences using a bar chart:

```python
sns.barplot(x='group', y='converted', data=df, ci=False)
```

---

### 3️⃣ Hypothesis Testing
**Null Hypothesis (H₀):**  pₐ = p_b  
**Alternative Hypothesis (H₁):**  pₐ ≠ p_b  

Used a **two-sample proportion Z-test**:

```python
p_pool = (success_a + success_b) / (n_a + n_b)
se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
z = (p_b - p_a) / se
p_value = 2 * (1 - stats.norm.cdf(abs(z)))
```

---

### 4️⃣ Results and interpretation

Compute conversion counts, rates, the Z-score, and the two-sided p-value by running the notebook against the committed CSV. The former example values were not a verified report of that file. A significant two-sided test indicates a difference; check its direction and effect size before recommending a variant. This project is documented as a simulation, so it does not support deploying a real website change.

---

## 📊 Visual Summary
- **Conversion Rate Bar Chart** comparing groups A and B.  
- **95 % Confidence Intervals** plotted to visualize uncertainty.  
- Optional: Tableau / Power BI dashboard for stakeholder presentation.

---

## 📁 Project Structure
```
ab_testing_project/
│
├── ab_test_results.csv         # dataset
├── ab_testing.ipynb      # notebook with EDA + testing
├── README_AB_Testing_Project.md # this file
└── (dependencies in ../requirements.txt)
```

---

## 🧾 Requirements
Install required libraries before running the notebook:

```bash
pip install pandas numpy seaborn matplotlib scipy jupyterlab
```

---

## 👤 Author
**David Solomon**  
Data Science & AI Learner  
📧 funomdave@gmail.com  

## Working directory

Run `ab_testing.ipynb` with this directory as the notebook working directory. Install `../requirements.txt` in the environment first.
