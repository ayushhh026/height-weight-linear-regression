# 📏 Height-Weight Linear Regression

> A simple linear regression project that predicts human height from weight using scikit-learn. Covers the full ML pipeline — EDA, feature scaling with StandardScaler, model training, and evaluation using MAE, RMSE, and R² — built on a height-weight dataset.

---

## 📁 Project Structure

```
height-weight-linear-regression/
│
├── height-weight.csv        # Dataset
├── regression.ipynb         # Main notebook
└── README.md
```

---

## 🔄 ML Pipeline

### 1. Exploratory Data Analysis (EDA)

Visualized the relationship between Weight and Height using a scatter plot and checked feature correlation using a heatmap.

**Scatter Plot — Height vs Weight**

<img width="714" height="541" alt="Scatter Plot" src="https://github.com/user-attachments/assets/0b6ec3ac-6e31-496d-a684-6addaa4cdc5f" />

**Correlation Heatmap**

<img width="655" height="525" alt="Heatmap" src="https://github.com/user-attachments/assets/42c14ee8-1c3e-4b0e-aebc-d4c2863c2699" />

> The heatmap confirms a **strong positive correlation** between Weight and Height — making Weight a strong predictor.

---

### 2. Feature Selection

| Variable | Role | Type |
|----------|------|------|
| Weight | Independent (X) | DataFrame (2D) |
| Height | Dependent (y) | Series (1D) |

---

### 3. Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```

- **75%** training data
- **25%** test data

---

### 4. Feature Scaling (Standardization)

Applied **Z-score standardization** using `StandardScaler` to convert Weight into a distribution with mean = 0 and SD = 1.

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit + Transform on train
X_test = scaler.transform(X_test)        # Only Transform on test (prevents data leakage)
```

> ⚠️ `fit_transform` is used only on training data. Using it on test data would cause **data leakage**.

---

### 5. Model Training

```python
regression = LinearRegression()
regression.fit(X_train, y_train)
```

The regression equation learned:

```
y = intercept + coef_ × X
y = 156.47 + 17.29 × X_standardized
```

- **Slope (coef_):** For every 1 unit increase in standardized weight, height increases by ~17.29 cm
- **Intercept:** When X = 0, predicted height = 156.47 cm

---

### 6. Best Fit Line

**Regression Line on Training Data**

<img width="689" height="515" alt="Best Fit Line" src="https://github.com/user-attachments/assets/4ed251ec-bbdf-465d-972b-b078af9093c7" />

---

### 7. Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| R² | Coefficient of Determination |
| Adjusted R² | R² adjusted for number of features |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

n, p = X_test.shape
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
```

---

### 8. Prediction on New Data

```python
# Always standardize new input before predicting
regression.predict(scaler.transform([[72]]))
```

> Raw input must be passed through the **same scaler** fitted on training data.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data-green?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Viz-red)
![Seaborn](https://img.shields.io/badge/Seaborn-Viz-9cf)

---

## 🚀 How to Run

```bash
git clone https://github.com/your-username/height-weight-linear-regression.git
cd height-weight-linear-regression
pip install pandas matplotlib seaborn scikit-learn numpy
jupyter notebook regression.ipynb
```

---

## 📌 Key Learnings

- Why we use **2D arrays** for independent features and **1D Series** for dependent
- Importance of **StandardScaler** and why `fit_transform` must not be used on test data
- How to interpret **slope and intercept** in the context of real data
- Difference between **R²** and **Adjusted R²**
