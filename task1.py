import os, zipfile, glob, pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# === 1. Unzip dataset ===
zip_path = "archive.zip"
extract_dir = "archive_extracted"
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)
    print("✅ Extracted:", zip_path)

# === 2. Load CSV ===
csv_files = glob.glob(os.path.join(extract_dir, "*.csv"))
df = pd.read_csv(csv_files[0])
print(df.head())

# === 3. Prepare data ===
target_col = "Species"
X = df.drop(columns=[target_col])
y = df[target_col]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 4. Train models ===
models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVC": SVC(probability=True, random_state=42)
}

results = {}
for name, model in models.items():
    if name in ("LogisticRegression", "SVC"):
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = (model, acc)
    print(f"{name} Accuracy: {acc:.4f}")

# === 5. Best model ===
best_model = max(results.items(), key=lambda x: x[1][1])
print("\nBest Model:", best_model[0])

# === 6. Evaluation ===
model = best_model[1][0]
preds = model.predict(X_test_scaled if best_model[0] != "RandomForest" else X_test)
print("\nClassification Report:\n", classification_report(y_test, preds, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

# === 7. Save model ===
artifact = {"model": model, "scaler": scaler, "label_encoder": le, "feature_columns": list(X.columns)}
with open("best_model.pkl", "wb") as f:
    pickle.dump(artifact, f)
print("✅ Model saved as best_model.pkl")

# === 8. Simple scatter plot ===
plt.scatter(df["SepalLengthCm"], df["SepalWidthCm"], c=y_encoded)
plt.xlabel("SepalLengthCm")
plt.ylabel("SepalWidthCm")
plt.title("Iris Species Visualization")
plt.show()
