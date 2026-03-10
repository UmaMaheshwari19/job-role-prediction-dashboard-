import streamlit as st
import pandas as pd
import pickle
import os

# -----------------------------
# Session State
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="Job Role Prediction", layout="wide")

# -----------------------------
# Custom UI Styling
# -----------------------------
st.markdown("""
<style>

/* Main background */
.stApp{
    background-color:#F3F4F6;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background-color:#0F172A;
}

/* Sidebar text */
section[data-testid="stSidebar"] *{
    color:white !important;
}

/* Sidebar title */
.sidebar-title{
    font-size:24px;
    font-weight:bold;
    text-align:center;
    margin-bottom:20px;
}

/* Buttons */
.stButton>button{
    background-color:#2563EB;
    color:white;
    border-radius:8px;
    height:40px;
    width:100%;
    font-size:16px;
}

.stButton>button:hover{
    background-color:#1D4ED8;
}

/* Headings */
h1,h2,h3{
    color:#1E293B;
}

/* Result Card */
.result-card{
    background-color:#22c55e;
    padding:20px;
    border-radius:10px;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")
data_path = os.path.join(BASE_DIR, "job_dataset.csv")
users_path = os.path.join(BASE_DIR, "users.csv")

# -----------------------------
# Load Model
# -----------------------------
with open(model_path, "rb") as f:
    model, le_degree, le_specialization, le_job = pickle.load(f)

df = pd.read_csv(data_path)

# -----------------------------
# User Functions
# -----------------------------
def load_users():
    return pd.read_csv(users_path)

def register_user(username, password):
    users = load_users()
    new_user = pd.DataFrame([[username, password]], columns=["username", "password"])
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv(users_path, index=False)

def login_user(username, password):
    users = pd.read_csv(users_path)

    users["username"] = users["username"].astype(str)
    users["password"] = users["password"].astype(str)

    user = users[(users["username"] == str(username)) & (users["password"] == str(password))]
    return not user.empty

# -----------------------------
# Title
# -----------------------------
st.title("🎓 AI Job Role Prediction System")

st.markdown(
"""
<h3 style='text-align:center;'>Predict the best career path based on your academic profile 🚀</h3>
""",
unsafe_allow_html=True
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown('<div class="sidebar-title">🚀 AI Career App</div>', unsafe_allow_html=True)

menu = ["🏠 Home","🔑 Login", "📝 Register"]
choice = st.sidebar.radio("Navigation", menu)

if choice == "🏠 Home":
    st.subheader("Welcome")
    st.write("This AI system predicts the best job role based on academic performance.")

# -----------------------------
# LOGIN PAGE
# -----------------------------
if choice == "🔑 Login":

    st.subheader("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if login_user(username, password):
            st.session_state.logged_in = True
            st.success("Login Successful")
        else:
            st.error("Invalid Username or Password")

# -----------------------------
# REGISTER PAGE
# -----------------------------
elif choice == "📝 Register":

    st.subheader("Create Account")

    new_user = st.text_input("Username")
    new_pass = st.text_input("Password", type="password")

    if st.button("Register"):
        register_user(new_user, new_pass)

        st.success("User Registered Successfully")
        st.info("Go to Login Page")

# -----------------------------
# AFTER LOGIN
# -----------------------------
if st.session_state.logged_in:

    st.sidebar.markdown("### Dashboard")

    page = st.sidebar.radio(
        "Select Page",
        ["📊 Prediction", "📂 Dataset Insights"]
    )

    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()
    # -----------------------------
    # PREDICTION
    # -----------------------------
    if page == "📊 Prediction":

        st.subheader("Enter Academic Details")
        
        st.metric("Model Accuracy", "87%")

        col1, col2 = st.columns(2)

        with col1:
            degree = st.selectbox("Degree", le_degree.classes_)
            specialization = st.selectbox("Specialization", le_specialization.classes_)

        with col2:
            cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.1)

        if st.button("Predict Job Role"):

            degree_encoded = le_degree.transform([degree])[0]
            specialization_encoded = le_specialization.transform([specialization])[0]

            prediction = model.predict([[degree_encoded, specialization_encoded, cgpa]])

            job = le_job.inverse_transform(prediction)[0]
            
            st.download_button(
    label="📥 Download Prediction Result",
    data=job,
    file_name="predicted_job_role.txt"
)

            st.markdown(
            f"""
            <div class="result-card">
            <h2 style="color:white;">🎯 Predicted Job Role</h2>
            <h1 style="color:white;">{job}</h1>
            </div>
            """,
            unsafe_allow_html=True
            )

    # -----------------------------
    # DATASET INSIGHTS
    # -----------------------------
    elif page == "📂 Dataset Insights":

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Job Role Distribution")

        job_counts = df["JobRole"].value_counts()

        st.bar_chart(job_counts)