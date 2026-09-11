from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ⚠️ สำคัญ — Normalize ก่อน KNN
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ===== 1. หาค่า K ที่ดีที่สุดด้วย Cross-Validation =====
k_range = range(1, 31)
cv_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    scores = cross_val_score(knn, X_train_s, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

best_k = k_range[np.argmax(cv_scores)]
print(f"✅ Best K = {best_k}, CV Accuracy = {max(cv_scores):.2%}")

plt.figure(figsize=(10,5))
plt.plot(k_range, cv_scores, marker='o', color='#2980b9')
plt.axvline(best_k, color='red', linestyle='--', label=f'Best K={best_k}')
plt.xlabel('K'); plt.ylabel('CV Accuracy'); plt.legend(); plt.grid(alpha=0.3); plt.show()

# ===== 2. Weighted KNN =====
knn_w = KNeighborsClassifier(n_neighbors=best_k, weights='distance', metric='euclidean')
knn_w.fit(X_train_s, y_train)
print(f"🌐 Weighted KNN Accuracy: {accuracy_score(y_test, knn_w.predict(X_test_s)):.2%}")

# ===== 3. ทดสอบหลาย Distance Metrics =====
print("\n📊 Distance Metrics เปรียบเทียบ:")
for metric in ['euclidean', 'manhattan', 'minkowski']:
    knn = KNeighborsClassifier(n_neighbors=best_k, metric=metric)
    scores = cross_val_score(knn, X_train_s, y_train, cv=5)
    print(f"  {metric:12s}: {scores.mean():.2%}")

# ===== 4. Grid Search (K + Weight + Metric) =====
param_grid = {'n_neighbors': list(range(1,21)), 'weights': ['uniform','distance'], 'metric': ['euclidean','manhattan']}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_s, y_train)
print(f"\n🔍 Best Params: {grid.best_params_}")
print(f"🎯 Best CV Accuracy: {grid.best_score_:.2%}")