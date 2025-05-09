import streamlit as st  
st.set_page_config( 
    page_title="Diabetes Detection App",  
    page_icon="🩺",  
    layout="centered"  
)

import pandas as pd  
import matplotlib.pyplot as plt  

# here we are importing the backend ml and actr models
from ml_model import rf_model, X, y, accuracy, conf_matrix, predict_diabetes, get_feature_importance
from actr_model import actr_model as actr_model, predict_diabetes_actr  

# App title and subtitle
st.title("🧠 Diabetes Prediction (ML + ACT-R)")
st.subheader("A Hybrid Model Powered by Machine Learning and Cognitive Science")
st.markdown("---")  

# here we are making all the inputs to be in the sidebars
st.sidebar.header("🔍 Enter Your Health Information")
# Taking user input from the sidebar
name = st.sidebar.text_input("Name")
sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
alcohol = st.sidebar.selectbox("High Alcohol Consumption?", ["No", "Yes"])
smoker = st.sidebar.selectbox("Smoker?", ["No", "Yes"])
highbp = st.sidebar.selectbox("High Blood Pressure?", ["No", "Yes"])
highchol = st.sidebar.selectbox("High Cholesterol?", ["No", "Yes"])
stroke = st.sidebar.selectbox("History of Stroke?", ["No", "Yes"])
physact = st.sidebar.selectbox("Physical Activity?", ["No", "Yes"])

# lastly, the very last button will be the prediction button
if st.sidebar.button("🔎 Predict Now"):
    # here we will collect the input data for the ML model
    input_data = pd.DataFrame([{
        'high_bp': 1 if highbp == "Yes" else 0,
        'highChol': 1 if highchol == "Yes" else 0,
        'BMI': bmi,
        'smoker': 1 if smoker == "Yes" else 0,
        'stroke': 1 if stroke == "Yes" else 0,
        'physicalactivity': 1 if physact == "Yes" else 0,
        'hvyalcohol_consumption': 1 if alcohol == "Yes" else 0,
        'Sex': 1 if sex == "Male" else 0
    }])

    #  the is the ML Prediction part
    ml_prediction = predict_diabetes(input_data)  
    ml_result = 'Diabetic' if ml_prediction == 1 else 'Non-Diabetic'

    # here we have the ACT-R Prediction 
    # here it tells the bmi function based on globally set numbers
    def bmi_group_func(bmi_val):
        if bmi_val < 25:
            return 'normal'
        elif bmi_val < 30:
            return 'overweight'
        else:
            return 'obese'
    # here we are preparing query for the ACT-R cognitive model
    actr_query = {
        'bmi_group': bmi_group_func(bmi),
        'smoker': 'yes' if smoker == "Yes" else 'no',
        'high_bp': 'yes' if highbp == "Yes" else 'no',
        'highChol': 'yes' if highchol == "Yes" else 'no',
        'stroke': 'yes' if stroke == "Yes" else 'no',
        'phys_activity': 'yes' if physact == "Yes" else 'no',
        'alcohol': 'yes' if alcohol == "Yes" else 'no',
        'sex': 'male' if sex == "Male" else 'female'
    }
    # here we are getting the result from the ACT-R model prediction
    actr_diagnosis, diabetic_percentage, warning_flag = predict_diabetes_actr(actr_model, actr_query)

    # Retrieve matching patients from ACT-R and count diabetics vs non-diabetics
    matching_patients = actr_model.retrieve_all_similar_patients(actr_query)
    total_matches = len(matching_patients)
    diabetic_count = sum(1 for p in matching_patients if p['diagnosis'] == 'yes')
    non_diabetic_count = total_matches - diabetic_count


    # # this part is for displaying result 
    # Show ML result
    st.success(f"✅ Machine Learning Prediction: {ml_result}")

    # here we arw ahowing ACTR result with counts
    st.info(
        f"🧠 ACT-R Cognitive Model Diagnosis: {actr_diagnosis.capitalize()} "
        f"({diabetic_percentage:.2f}% diabetic probability)\n"
        f"🧾 Matching Patients Found: {total_matches}\n"
        f" Diabetic: {diabetic_count}\n"
        f" Non-Diabetic: {non_diabetic_count}"
)


    if warning_flag:
        st.warning("⚠ Very few matching patients found. Diagnosis may be less reliable.")

    # Redirect to healthcare if diabetic
    if ml_result == "Diabetic" or actr_diagnosis.lower() in ["mostly diabetic", "diabetic"]:
        st.markdown("[🏥 Visit Walgreens Virtual HealthCare](https://www.walgreens.com/topic/virtual-healthcare.jsp)", unsafe_allow_html=True)

    # here we are showing Random Forest accuracy
    st.markdown("---")
    st.subheader("📊 Model Performance Overview")
    st.write(f"Random Forest Model Accuracy: {accuracy:.2f}")

    # below we are showing how features are prioritized by our ML model
    st.subheader("📈 Feature Importance (from ML Model)")
    feature_names, importances = get_feature_importance()
    # here we are plotting the feature importance graph
    fig, ax = plt.subplots()
    ax.barh(feature_names, importances, color="#0B5394")  
    plt.xlabel("Importance")
    plt.title("Features Contributing to Prediction")
    st.pyplot(fig)

# this is a footer for fun
st.markdown("---")
st.caption("© 2025 Health-2 Project 🚀")


# Logos of Penn State and Google Colab (centered and resized properly)
st.markdown("---")
st.subheader("🌐 Powered By:")

st.markdown("""
<div style='text-align: center;'>
    <img src='https://i.imgur.com/nNUcvMs.png' width='220' style='margin-right: 40px;'/>
    <img src='https://i.imgur.com/rLPe08g.png' width='220'/>
</div>
""", unsafe_allow_html=True)
