# ckd_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from boruta import BorutaPy

# Step 1: Load dataset
df = pd.read_csv('kidney_disease.csv')

# Step 2: Preprocess
# Handle missing values and encode categorical variables
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna('Unknown')
        df[col] = LabelEncoder().fit_transform(df[col])
    else:
        df[col] = df[col].fillna(df[col].median())

# Step 3: Feature/target split
X = df.drop(['classification','id'] ,axis=1)  # assuming 'class' is the target column
y = df['classification']

# Step 4: Feature selection with Boruta
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=42)
boruta_selector = BorutaPy(estimator=rf, n_estimators='auto', verbose=1, random_state=42)
boruta_selector.fit(X.values, y.values)

# Step 5: Get selected features
selected_features = X.columns[boruta_selector.support_].to_list()
print("Selected features:", selected_features)

# Step 6: Train/test split with selected features
X_selected = X[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Step 7: Scale the data (important for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 8: Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)  # you can tune this
knn.fit(X_train_scaled, y_train)

# Step 9: Evaluate
y_pred = knn.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))