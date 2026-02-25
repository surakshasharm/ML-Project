import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ==============================
# 1. Load Dataset
# ==============================
df = pd.read_csv("SeoulBikeData.csv", encoding="ISO-8859-1")

df.columns = ['Date', 'Rented_Bike_Count', 'Hour', 'Temperature', 'Humidity',
              'Wind_Speed', 'Visibility', 'Dew_Point_Temp', 'Solar_Radiation',
              'Rainfall', 'Snowfall', 'Seasons', 'Holiday', 'Functioning_Day']

df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
print(f"Data Loaded: {df.shape[0]} rows, {df.shape[1]} columns")


# ==============================
# 2. EDA
# ==============================
plt.figure(figsize=(10, 5))
sns.histplot(df['Rented_Bike_Count'], kde=True)
plt.title("Distribution of Rented Bike Count")
plt.show()

plt.figure(figsize=(10, 5))
sns.lineplot(x="Hour", y="Rented_Bike_Count", data=df, hue="Seasons", marker="o")
plt.title("Bike Demand by Hour and Season")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()


# ==============================
# 3. Preprocessing
# ==============================
X = df.drop(["Date", "Rented_Bike_Count"], axis=1)
y_reg = df["Rented_Bike_Count"]

categorical_cols = ["Seasons", "Holiday", "Functioning_Day"]
numerical_cols = ["Hour", "Temperature", "Humidity", "Wind_Speed", "Visibility",
                  "Dew_Point_Temp", "Solar_Radiation", "Rainfall", "Snowfall"]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])


# ==============================
# 4. Regression Models + PCA
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

print("\n✅ Linear Regression + PCA")
lr_pca_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("pca", PCA(n_components=0.95)),  # Keep 95% variance
    ("model", LinearRegression())
])

lr_pca_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pca_pipeline.predict(X_test)

print(f"R2 Score (Linear Regression + PCA): {r2_score(y_test, y_pred_lr):.4f}")


print("\n✅ Random Forest Regressor (without PCA)")
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=200, random_state=42))
])

rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)

print(f"R2 Score (Random Forest): {r2_score(y_test, y_pred_rf):.4f}")

plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred_rf, alpha=0.3)
plt.plot([0, max(y_test)], [0, max(y_test)], "r--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Random Forest: Actual vs Predicted")
plt.show()


# ==============================
# 5. Hyperparameter Tuning (Random Forest)
# ==============================
print("\n✅ Hyperparameter Tuning using GridSearchCV...")

rf_grid = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    rf_grid, param_grid, cv=3, scoring="r2", n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best Params:", grid_search.best_params_)
best_rf = grid_search.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)

print(f"R2 Score (Tuned RF): {r2_score(y_test, y_pred_best_rf):.4f}")


# ==============================
# 6. KMeans Clustering + Elbow + Silhouette
# ==============================
print("\n✅ KMeans Clustering (Advanced)")

weather_features = df[["Temperature", "Humidity", "Wind_Speed", "Rainfall"]]
weather_scaled = StandardScaler().fit_transform(weather_features)

inertia = []
K_range = range(2, 10)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(weather_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(K_range, inertia, marker="o")
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Weather_Cluster"] = kmeans.fit_predict(weather_scaled)

sil_score = silhouette_score(weather_scaled, df["Weather_Cluster"])
print(f"Silhouette Score (K=4): {sil_score:.4f}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x="Temperature", y="Humidity", hue="Weather_Cluster", data=df, palette="viridis")
plt.title("K-Means Clustering of Weather Conditions")
plt.show()


# ==============================
# 7. Classification Task
# ==============================
threshold = 700
df["High_Demand"] = (df["Rented_Bike_Count"] > threshold).astype(int)

y_cls = df["High_Demand"]
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_cls, test_size=0.2, random_state=42
)

print("\n✅ Logistic Regression + PCA")
log_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("pca", PCA(n_components=0.95)),
    ("model", LogisticRegression(max_iter=2000))
])

log_pipeline.fit(X_train_c, y_train_c)
y_pred_log = log_pipeline.predict(X_test_c)

print(f"Accuracy (LogReg + PCA): {accuracy_score(y_test_c, y_pred_log):.4f}")
print("\nClassification Report:\n", classification_report(y_test_c, y_pred_log))


print("\n✅ Random Forest Classifier (More Advanced)")
rf_cls_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=200, random_state=42))
])

rf_cls_pipeline.fit(X_train_c, y_train_c)
y_pred_rfcls = rf_cls_pipeline.predict(X_test_c)

print(f"Accuracy (RF Classifier): {accuracy_score(y_test_c, y_pred_rfcls):.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test_c, y_pred_rfcls))


# ==============================
# 8. Decision Tree Visualization
# ==============================
print("\n✅ Decision Tree Classifier")
dt_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", DecisionTreeClassifier(max_depth=3, random_state=42))
])

dt_pipeline.fit(X_train_c, y_train_c)
y_pred_dt = dt_pipeline.predict(X_test_c)

print(f"Accuracy (Decision Tree): {accuracy_score(y_test_c, y_pred_dt):.4f}")

plt.figure(figsize=(15, 8))
feature_names = (numerical_cols +
                 list(dt_pipeline.named_steps["preprocessor"]
                      .transformers_[1][1].get_feature_names_out(categorical_cols)))

plot_tree(dt_pipeline.named_steps["model"], feature_names=feature_names, filled=True, fontsize=9)
plt.title("Decision Tree for High Bike Demand")
plt.show()

