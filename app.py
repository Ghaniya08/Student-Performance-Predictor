import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Student Performance Predictor', layout='centered')

@st.cache_resource
def load_model():
    return joblib.load('student_model.joblib')

st.title('Student Performance Predictor')
st.write('Predict **Final Exam Score** and **Pass / Fail**')

try:
    bundle = load_model()
except FileNotFoundError:
    st.error('Model missing. Run `python train_model.py` first.')
    st.stop()

study_hours = st.number_input('Study Hours per Week', min_value=0, max_value=60, value=25)
attendance = st.number_input('Attendance Rate (%)', min_value=0.0, max_value=100.0, value=80.0)
past_scores = st.number_input('Past Exam Scores', min_value=0, max_value=100, value=75)
gender = st.selectbox('Gender', list(bundle['encoders']['Gender'].classes_))
parental = st.selectbox('Parental Education Level', list(bundle['encoders']['Parental_Education_Level'].classes_))
internet = st.selectbox('Internet Access at Home', list(bundle['encoders']['Internet_Access_at_Home'].classes_))
extracurricular = st.selectbox('Extracurricular Activities', list(bundle['encoders']['Extracurricular_Activities'].classes_))

if st.button('Predict'):
    enc = bundle['encoders']
    row = {
        'Study_Hours_per_Week': study_hours,
        'Attendance_Rate': attendance,
        'Past_Exam_Scores': past_scores,
        'Gender': enc['Gender'].transform([gender])[0],
        'Parental_Education_Level': enc['Parental_Education_Level'].transform([parental])[0],
        'Internet_Access_at_Home': enc['Internet_Access_at_Home'].transform([internet])[0],
        'Extracurricular_Activities': enc['Extracurricular_Activities'].transform([extracurricular])[0],
    }
    X = pd.DataFrame([row])[bundle['feature_cols']]
    score = int(np.clip(np.round(bundle['model'].predict(X)[0]), 0, 100))
    result = 'Pass' if score >= bundle['passing_marks'] else 'Fail'

    color = 'green' if result == 'Pass' else 'red'
    st.markdown(f'### Final Exam Score: **{score}**')
    st.markdown(f'### Result: :{color}[{result}]')

with st.expander('Model Info (Accuracy / Metrics)'):
    m = bundle.get('metrics', {})
    st.write(f"**Model:** {bundle.get('model_name', 'RandomForestRegressor')}")
    st.write(f"**Passing Marks:** {bundle.get('passing_marks', 60)}")
    if m:
        st.write(f"**Accuracy:** {m.get('pass_fail_accuracy', 0)*100:.2f}%")
        st.write(f"**R2 Score:** {m.get('r2', 0):.3f}")
        st.write(f"**MAE:** {m.get('mae', 0):.2f}")
    else:
        st.write('Metrics not found. Run `python train_model.py` again.')
