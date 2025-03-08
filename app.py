import streamlit as st
import os
from signature_verification import store_signature_features, verify_signature

# Paths
REAL_SIGNATURES_DIR = "real_signatures"
FORGED_SIGNATURES_DIR = "forged_signatures"

os.makedirs(REAL_SIGNATURES_DIR, exist_ok=True)
os.makedirs(FORGED_SIGNATURES_DIR, exist_ok=True)

# UI Design
st.title("Signature Verification System")

# User Registration
st.sidebar.header("User Registration")
user_id = st.sidebar.text_input("Enter User ID:")
pin = st.sidebar.text_input("Enter PIN:", type="password")

if st.sidebar.button("Register"):
    user_data_path = f"user_data/{user_id}.txt"
    if os.path.exists(user_data_path):
        st.sidebar.warning("User already exists! Try a different ID.")
    else:
        with open(user_data_path, "w") as f:
            f.write(pin)
        st.sidebar.success("User Registered Successfully!")

# Upload Signatures
st.header("Upload Your Signatures")

uploaded_files = st.file_uploader("Upload at least 5 genuine signatures", accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(REAL_SIGNATURES_DIR, f"{user_id}_{file.name}")
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        store_signature_features(user_id, file_path)
    st.success("✅ Signatures stored successfully!")

# Signature Verification
st.header("Verify Your Signature")
input_file = st.file_uploader("Upload a signature for verification")

if input_file:
    input_path = os.path.join(FORGED_SIGNATURES_DIR, f"{user_id}_input.png")
    with open(input_path, "wb") as f:
        f.write(input_file.getbuffer())

    result = verify_signature(user_id, input_path)
    st.write("🔍 Verification Result:", result)
