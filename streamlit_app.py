import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

st.markdown("""
<style>

/* Full Background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Center Card */
.main {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(15px);
    padding: 40px;
    border-radius: 20px;
    width: 70%;
    margin: auto;
    box-shadow: 0px 8px 32px 0 rgba(0, 0, 0, 0.37);
}

/* Labels */
label {
    color: #ffffff !important;
    font-weight: 600;
    font-size: 16px;
}

/* Input Fields */
.stSelectbox div div,
.stNumberInput input {
    background-color: rgba(255, 255, 255, 0.1) !important;
    color: white !important;
    border-radius: 10px;
}

/* Button Style */
.stButton>button {
    background: linear-gradient(90deg, #00C9A7, #92FE9D);
    color: black;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 17px;
    font-weight: bold;
    border: none;
    transition: 0.3s ease;
}

/* Button Hover Glow */
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0px 0px 20px #00C9A7;
}

/* Title Styling */
h1 {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: #ffffff;
}

</style>
""", unsafe_allow_html=True)

# Load model
with open("model.pkl", "rb") as f:
    model, le_degree, le_specialization, le_job = pickle.load(f)

# Load dataset
df = pd.read_csv("job_dataset.csv")

# Page config
st.set_page_config(page_title="Job Role Prediction", layout="centered")

# Title
st.title("🎓 Job Role Prediction Dashboard")

st.markdown("---")

# Input fields
degree = st.selectbox("Select Degree", le_degree.classes_)
specialization = st.selectbox("Select Specialization", le_specialization.classes_)
cgpa = st.number_input("Enter CGPA", min_value=0.0, max_value=10.0, step=0.1)

# Prediction
if st.button("Predict Job"):
    deg_encoded = le_degree.transform([degree])[0]
    spec_encoded = le_specialization.transform([specialization])[0]

    prediction = model.predict([[deg_encoded, spec_encoded, cgpa]])
    result = le_job.inverse_transform(prediction)[0]

    st.success(f"Predicted Job Role: {result}")

st.markdown("---")

job_counts = df["JobRole"].value_counts()

fig, ax = plt.subplots(figsize=(10, 6))

job_counts.plot(kind="barh", ax=ax)   # horizontal bar chart

plt.xlabel("Count")
plt.ylabel("Job Role")
plt.tight_layout()

st.pyplot(fig)