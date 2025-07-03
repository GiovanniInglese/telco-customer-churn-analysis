import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# === 2. Load Dataset ===
df = pd.read_csv("telco_churn_cleaned.csv")
# Ensure churn_label is binary (0 = No, 1 = Yes)
df['churn_label'] = df['churn_label'].str.lower().map({'no': 0, 'yes': 1})


#Prepare Features 
y = df['churn_label']                    # Already binary: 0 or 1
X = df.drop(['churn_label'], axis=1)     # Drop target from features

# One-hot encode if needed (safe if already encoded)
X = pd.get_dummies(X, drop_first=True)

#Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression (Baseline Model)
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print("\n=== Logistic Regression Report ===")
print(classification_report(y_test, y_pred_log))

'''
This logistic regression model more so better for identifying customers who stay or have not churned, 
the amount of actual curners caught by the mdoel was 59%, 
which means an additional 41% of churners were not caught.  
This could be bad for a business model so we will try another model and balance the dataset with SMOTE or  
synthetic minority oversampling technique which creates synthetic samples based on the minority class of the data set to balance the results
'''


# Random Forest with SMOTE 
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_res, y_train_res)
y_pred_rf = rf_model.predict(X_test)

print("\n=== Random Forest (SMOTE) Report ===")
print(classification_report(y_test, y_pred_rf))

'''This RandomForest model which utilizes decision trees predicted a little
better than the previous model with the recall  increasing from 59% to 60%
which means it slightly catches more real churners.  The precision decresed by 3% fro  68% to 65% 
but this is exceptable because we want to catch early churn.  The f1 score or balane between recall and precision
it remains at 62% which is good for an imbalance classification issue.


'''




# XGBoost Model (Best Performer)
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_res, y_train_res)
y_pred_xgb = xgb_model.predict(X_test)

print("\n=== XGBoost (SMOTE) Report ===")
print(classification_report(y_test, y_pred_xgb))

'''The XGboost model outperforms the other models and gives a higher precision score of 66% while maintaining the recall score of 66% and having a higher f1 score of 63%.
We can confirm that this is the more accurate and effective model for predicting customer churn at telco
'''



# === 8. Confusion Matrix for XGBoost ===
cm = confusion_matrix(y_test, y_pred_xgb)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stayed", "Churned"])
disp.plot(cmap="Blues")
plt.title("XGBoost Confusion Matrix")
plt.show()

# Model Explanation:
'''
The Confusion matrix represents True and false negatives and positives.  
Top Left- True negatives, correctly predicted customers who stayed
Top Right- False Positives, predicted churned customers that actually stayed
Bottom Left- False negatives, predicted stayed customers that actually churned
Bottom Right- True positives, correctly predicted churned customers.  
'''



# Export final results for Tableau
df_export = df.copy()  # Use the original dataframe

# Add predictions
df_export['predicted_churn'] = xgb_model.predict(X)

# Optional: readable labels
df_export['churn_label'] = df_export['churn_label'].apply(lambda x: 'Yes' if x == 1 else 'No')
df_export['predicted_label'] = df_export['predicted_churn'].apply(lambda x: 'Yes' if x == 1 else 'No')

# Export only useful columns for Tableau
columns_to_export = [
    'tenure', 'MonthlyCharges', 'TotalCharges',  # numerical predictors
    'Contract', 'PaymentMethod', 'InternetService', 'TechSupport',  # categorical predictors
    'churn_label', 'predicted_label', 'churn_label', 'predicted_churn'
]

# Filter to just those columns (if they exist)
df_export = df_export[[col for col in columns_to_export if col in df_export.columns]]

# Export to clean CSV
#df_export.to_csv("churn_model_results.csv", index=False)

