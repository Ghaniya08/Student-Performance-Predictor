import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score

PASSING_MARKS = 60

df = pd.read_csv('student_performance_dataset.csv')
df.head()

df.shape

# encode categorical columns
cat_cols = ['Gender','Parental_Education_Level','Internet_Access_at_Home','Extracurricular_Activities']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

X = df[['Study_Hours_per_Week','Attendance_Rate','Past_Exam_Scores',
        'Gender','Parental_Education_Level','Internet_Access_at_Home','Extracurricular_Activities']]
y = df['Final_Exam_Score']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print(X_train.shape)
print(X_test.shape)

rf = RandomForestRegressor()
rf.fit(X_train,y_train)
y_pred = np.clip(np.round(rf.predict(X_test)),0,100)
print('R2:', r2_score(y_test,y_pred))
print('MAE:', mean_absolute_error(y_test,y_pred))
print('Pass/Fail Acc:', accuracy_score((y_test>=PASSING_MARKS).astype(int),(y_pred>=PASSING_MARKS).astype(int)))

rf = RandomForestRegressor(max_samples=0.75,random_state=42)
rf.fit(X_train,y_train)
y_pred = np.clip(np.round(rf.predict(X_test)),0,100)

r2 = r2_score(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
pass_acc = accuracy_score((y_test>=PASSING_MARKS).astype(int),(y_pred>=PASSING_MARKS).astype(int))

print('RF max_samples=0.75 R2:', r2)
print('RF max_samples=0.75 MAE:', mae)
print('RF max_samples=0.75 Pass/Fail Acc:', pass_acc)

joblib.dump({
    'model': rf,
    'encoders': encoders,
    'feature_cols': list(X.columns),
    'model_name': 'RandomForestRegressor',
    'passing_marks': PASSING_MARKS,
    'metrics': {
        'r2': float(r2),
        'mae': float(mae),
        'pass_fail_accuracy': float(pass_acc)
    },
}, 'student_model.joblib')
print('Saved: student_model.joblib')
