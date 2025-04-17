import numpy as np
import streamlit as st
from joblib import load
from PIL import Image
from pathlib import Path
import gdown

# File paths
MODEL_PATH = Path("Trained_Diabetes_model.joblib")
IMAGE_PATH = Path("img.jpeg")

# Google Drive file IDs (replace with actual file IDs)
MODEL_FILE_ID = "1xojGeZcX1tMlGbJtjNOM86869qBvT4AQ"
IMAGE_FILE_ID = "1ap7DDQY3TmVvOk7Il60kVu33q_CGreu7"

# Download files from Google Drive if they don't exist
def download_if_missing(file_path, file_id):
    if not file_path.exists():
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(file_path), quiet=False)

# Use Streamlit's cache to load model only once
@st.cache_resource
def load_model():
    download_if_missing(MODEL_PATH, MODEL_FILE_ID)
    return load(MODEL_PATH)

# Load the model
model = load_model()

# Prediction function
def diabetes_prediction(input_data):
    try:
        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)
        return "You have diabetes" if prediction[0] == 1 else "You do not have diabetes"
    except Exception as e:
        return f"Error making prediction: {e}"

def main():
    st.title("Diabetes Prediction App")

    # Load and display image
    try:
        download_if_missing(IMAGE_PATH, IMAGE_FILE_ID)
        with Image.open(IMAGE_PATH) as img:
            st.image(img.resize((200, 200)), width=200)
    except Exception:
        st.warning("Image could not be loaded.")

    st.sidebar.title("Input Features")

    HighBP = st.sidebar.radio("High Blood Pressure", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    HighChol = st.sidebar.radio("High Cholesterol", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    BMI = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    Smoker = st.sidebar.radio("Smoker (100+ cigarettes in life)", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    Fruits = st.sidebar.radio("Eat fruit 1+ times per day?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    GenHlth = st.sidebar.selectbox("General Health Rating", [1, 2, 3, 4, 5],
                                   format_func=lambda x: ["Excellent", "Very Good", "Good", "Fair", "Poor"][x - 1])
    MentHlth = st.sidebar.slider("Poor mental health days (last 30)", 0, 30, 5)
    PhysHlth = st.sidebar.slider("Poor physical health days (last 30)", 0, 30, 5)
    Sex = st.sidebar.radio("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")

    Age = st.sidebar.selectbox("Age Group", list(range(1, 14)),
                               format_func=lambda x: ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
                                                      "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"][x - 1])

    Education = st.sidebar.selectbox("Education", list(range(1, 7)),
                                     format_func=lambda x: ["No Formal Education", "Elementary School", "High School Student",
                                                            "High School Graduate", "College Student", "College Graduate"][x - 1])
    Income = st.sidebar.selectbox("Income", list(range(1, 9)),
                                  format_func=lambda x: ["below $10K", "$10K-$14.9K", "$15K-$19.9K", "$20K-$24.9K",
                                                         "$25K-$34.9K", "$35K-$49.9K", "$50K-$74.9K", "$75K+"][x - 1])

    if st.sidebar.button("Predict"):
        features = [HighBP, HighChol, BMI, Smoker, Fruits, GenHlth, MentHlth, PhysHlth,
                    Sex, Age, Education, Income]
        result = diabetes_prediction(features)

        st.subheader("Prediction Result")
        if "not diabetic" in result:
            st.success(result)
        elif "diabetic" in result:
            st.warning(result)
        else:
            st.error(result)

if __name__ == "__main__":
    main()
