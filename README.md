# ML-Project
Seoul Bike Sharing dataset is analyzed to predict bike rental demand. EDA is done using graphs and correlation heatmap. Regression models (Linear Regression, Random Forest) predict bike count. K-Means clusters weather patterns. Logistic Regression and Decision Tree classify high-demand rentals with accuracy and reports.

# 🚲 Seoul Bike Rental Demand Prediction (ML Project)

## 📌 Project Overview
This project analyzes the Seoul Bike Sharing dataset and builds Machine Learning models to predict bike rental demand. It includes data visualization, regression for bike count prediction, clustering of weather conditions, and classification of high-demand hours.

---

## 📂 Dataset
- File: `SeoulBikeData.csv`
- Target Variable: **Rented_Bike_Count**
- Features include:
  - Hour, Temperature, Humidity, Wind Speed, Visibility
  - Dew Point Temp, Solar Radiation, Rainfall, Snowfall
  - Seasons, Holiday, Functioning Day

---

## ✅ Tasks Performed
### 🔍 Exploratory Data Analysis (EDA)
- Demand distribution plot
- Hourly demand trend by seasons
- Correlation heatmap for numerical features

### 📈 Regression (Bike Count Prediction)
- Linear Regression
- Random Forest Regressor  
Evaluation: **R² Score**

### 🌦️ Clustering (Weather Pattern Analysis)
- K-Means clustering on weather features  
(Temperature, Humidity, Wind Speed, Rainfall)

### 🧠 Classification (High Demand Prediction)
- Created a new label `High_Demand` using a threshold
- Logistic Regression
- Decision Tree Classifier  
Evaluation: Accuracy, Confusion Matrix, Classification Report

---

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn


