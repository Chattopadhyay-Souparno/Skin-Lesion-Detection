import numpy as np
import xgboost as xgb
import joblib
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix ,cohen_kappa_score
import pandas as pd


X_test = np.load('X_test.npy')  
X_test_1 = np.load('X_test_no_segmentation_hair.npy')  
X_val = np.load('X_val.npy')  
X_val_1 = np.load('X_val_no_segmentation_hair.npy')  
y_val = np.load('y_val.npy')  


if X_val.shape[0] != X_val_1.shape[0]:
    min_samples = min(X_val.shape[0], X_val_1.shape[0])
    X_val = X_val[:min_samples]
    X_val_1 = X_val_1[:min_samples]
    y_val = y_val[:min_samples]


base_model_1 = joblib.load('xgb_clf1_model.pkl')  
base_model_2 = joblib.load('Binary_models/xgb_clf2_model.pkl')  
meta_model = joblib.load('Binary_models/meta_model.pkl')  


val_meta_features1 = base_model_1.predict_proba(X_val)[:, 1].reshape(-1, 1)  
val_meta_features2 = base_model_2.predict_proba(X_val_1)[:, 1].reshape(-1, 1)  

X_val_meta = np.hstack((val_meta_features1, val_meta_features2))

val_predictions = meta_model.predict(X_val_meta)
val_accuracy = accuracy_score(y_val, val_predictions)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

val_kappa = cohen_kappa_score(y_val, val_predictions)
print(f"\nValidation Kappa (Stacked Model): {val_kappa:.2f}")

print("\nClassification Report:")
print(classification_report(y_val, val_predictions))
print("\nConfusion Matrix:")
print(confusion_matrix(y_val, val_predictions))




test_meta_features1 = base_model_1.predict_proba(X_test)[:, 1].reshape(-1, 1)
test_meta_features2 = base_model_2.predict_proba(X_test_1)[:, 1].reshape(-1, 1)


X_test_meta = np.hstack((test_meta_features1, test_meta_features2))


final_predictions = meta_model.predict(X_test_meta)


output_df = pd.DataFrame(final_predictions)


output_df.to_excel('test_predictions.xlsx', index=False, header=False)

print("Predictions saved to 'test_predictions.xlsx' with one column of classification results.")