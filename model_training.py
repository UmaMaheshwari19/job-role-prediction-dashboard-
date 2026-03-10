import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 🔹 Get current file directory safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "job_dataset.csv")

print("Reading file from:", file_path)

# 🔹 Load dataset
df = pd.read_csv(file_path)

# 🔹 Encode categorical columns
le_degree = LabelEncoder()
le_specialization = LabelEncoder()
le_job = LabelEncoder()

df["Degree"] = le_degree.fit_transform(df["Degree"])
df["Specialization"] = le_specialization.fit_transform(df["Specialization"])
df["JobRole"] = le_job.fit_transform(df["JobRole"])

# 🔹 Features and target
X = df[["Degree", "Specialization", "CGPA"]]
y = df["JobRole"]

# 🔹 Train model
model = RandomForestClassifier()
model.fit(X, y)

# 🔹 Save model
model_path = os.path.join(BASE_DIR, "model.pkl")

with open(model_path, "wb") as f:
    pickle.dump((model, le_degree, le_specialization, le_job), f)

print("✅ Model trained and saved successfully!")