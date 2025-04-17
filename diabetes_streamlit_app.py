import numpy as np
import streamlit as st
from joblib import load
from PIL import Image

# Use Streamlit's caching to avoid reloading model every time
@st.cache_resource
def load_model():
    return load(r"Trained_Diabetes_model.joblib")

# Load model only once
model = load_model()

# Prediction function
def diabetes_prediction(input_data):
    try:
        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)
        return "The person is diabetic" if prediction[0] == 1 else "The person is not diabetic"
    except Exception as e:
        return f"Error making prediction: {e}"

def main():
    st.title("Diabetes Prediction App")

    # Efficient image loading
    try:
        with Image.open(r"img.jpeg") as img:
            img = img.resize((200, 200))
            st.image(img, width=200)
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
                                     format_func=lambda x: ["No School", "Elementary", "Some High School",
                                                            "High School Grad", "Some College", "College Grad"][x - 1])
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
