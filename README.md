# Student-Performance-Predictor

This project is a **Student Performance Predictor** developed as a practice assignment for the Machine Learning Final Project. It utilizes a regression model to estimate student marks based on their study habits, attendance, and past performance.

## 📊 Model Information & Metrics

* **Algorithm:** `RandomForestRegressor`
* **Model Accuracy:** 88.73% (`0.8873239436619719`)
* **R² Score:** 0.778
* **Mean Absolute Error (MAE):** 2.12
* **Passing Marks Threshold:** 60

## ⚙️ Input Constraints & Validation

To ensure accurate predictions, the model enforces the following constraints on the input features:
* **Study Hours:** Must be $\le 60$ hours.
* **Attendance:** Must be $\le 100\%$.
* **Past Exam Score:** Must be $\le 100$ marks.

## 🚀 Getting Started

## Installation

To run this project, clone the repository and install the required dependencies using `pip`:

```bash
1. Navigate to the project directory
cd Student-Performance-Predictor

2. Install the required packages from requirements.txt
pip install -r requirements.txt
```

### Prerequisites
Make sure you have the following Python libraries installed:
```bash
pip install -r requirements.txt
```

## Dataset

The default dataset **`student_performance_dataset.csv`** is already uploaded and available in this GitHub repository. 

* **Using the default data:** If you want to test the model with the existing data, you can use the provided file directly without any changes.
* **Using your own data:** If you prefer to train or test the model with your own data, you can replace this file with your custom dataset. Just ensure your new data strictly follows the same column structure and the constraints mentioned above.


### Usage
1. Clone this repository.
3. Train or load the `RandomForestRegressor` model to predict whether a student will pass (score $\ge 60$) based on their hours, attendance, and past records.
