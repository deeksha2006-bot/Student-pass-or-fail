Student Placement Eligibility Prediction

Project Overview

This project predicts whether a student is eligible for placement using Machine Learning.

🎯 Objective

Build a binary classification model to predict student placement eligibility based on academic and experience-related factors.

📊 Dataset

- 1,000 student records
- 6 input features
- Target variable: "target"
- "target = 1" → Eligible
- "target = 0" → Not Eligible

Input Features

- "cgpa"
- "attendance_pct"
- "coding_score"
- "projects_completed"
- "internship_months"
- "backlogs"

🤖 Machine Learning Algorithm

Logistic Regression

Preprocessing:

- StandardScaler
- 80% training data
- 20% testing data
- Stratified train/test split

📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

📊 Visualizations

- Confusion Matrix
- ROC Curve
- Feature Coefficient Chart
- Target Class Distribution

🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib

💡 Conclusion

The project demonstrates how Logistic Regression can be used to predict student placement eligibility and identify the factors associated with placement outcomes.
