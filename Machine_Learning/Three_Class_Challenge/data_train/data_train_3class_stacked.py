import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd


X_train = np.load('X_train_3.npy')
y_train = np.load('y_train_3.npy')
X_val = np.load('X_val_3.npy')
y_val = np.load('y_val_3.npy')

X_train_1 = np.load('X_train_3_segmentation.npy')
y_train_1 = np.load('y_train_3_segmentation.npy')
X_val_1 = np.load('X_val_3_segmentation.npy')
y_val_1 = np.load('y_val_3_segmentation.npy')


X_train_1, y_train_1 = resample(X_train_1, y_train_1, n_samples=len(y_train), random_state=42)
X_val_1, y_val_1 = resample(X_val_1, y_val_1, n_samples=len(y_val), random_state=42)


kappa_scorer = make_scorer(cohen_kappa_score)


xgb_clf_1 = xgb.XGBClassifier(objective='multi:softprob', n_estimators=400, eval_metric='mlogloss', num_class=3, random_state=42)
xgb_clf_2 = xgb.XGBClassifier(objective='multi:softprob', n_estimators=400, eval_metric='mlogloss', num_class=3, random_state=42)


xgb_clf_1.fit(X_train, y_train)
xgb_clf_2.fit(X_train_1, y_train_1)


y_val_pred_1 = xgb_clf_1.predict(X_val)
y_val_pred_2 = xgb_clf_2.predict(X_val_1)
val_kappa_1 = cohen_kappa_score(y_val, y_val_pred_1)
val_kappa_2 = cohen_kappa_score(y_val_1, y_val_pred_2)
print(f"Validation Kappa (Base Model 1): {val_kappa_1:.2f}")
print(f"Validation Kappa (Base Model 2): {val_kappa_2:.2f}")


train_meta_features1 = xgb_clf_1.predict_proba(X_train)
train_meta_features2 = xgb_clf_2.predict_proba(X_train_1)
X_train_meta = np.hstack((train_meta_features1, train_meta_features2))


val_meta_features1 = xgb_clf_1.predict_proba(X_val)
val_meta_features2 = xgb_clf_2.predict_proba(X_val_1)
X_val_meta = np.hstack((val_meta_features1, val_meta_features2))


meta_clf = LogisticRegression(multi_class='multinomial', random_state=42, max_iter=200)
meta_clf.fit(X_train_meta, y_train)


y_val_pred_meta = meta_clf.predict(X_val_meta)
val_kappa_meta = cohen_kappa_score(y_val, y_val_pred_meta)
print(f"\nValidation Kappa (Stacked Model): {val_kappa_meta:.2f}")
print("\nClassification Report (Meta-Model):")
print(classification_report(y_val, y_val_pred_meta))
print("\nConfusion Matrix (Meta-Model):")
print(confusion_matrix(y_val, y_val_pred_meta))


joblib.dump(xgb_clf_1, 'xgb_clf1_model_3class.pkl')
joblib.dump(xgb_clf_2, 'xgb_clf2_model_3class.pkl')
joblib.dump(meta_clf, 'meta_model_3class.pkl')


X_test = np.load('X_test_3.npy')
X_test_1 = np.load('X_test_3_segmentation.npy')


base_model_1 = joblib.load('xgb_clf1_model_3class.pkl')
base_model_2 = joblib.load('xgb_clf2_model_3class.pkl')
meta_model = joblib.load('meta_model_3class.pkl')


test_meta_features1 = base_model_1.predict_proba(X_test)
test_meta_features2 = base_model_2.predict_proba(X_test_1)
X_test_meta = np.hstack((test_meta_features1, test_meta_features2))


final_predictions = meta_model.predict(X_test_meta)


output_df = pd.DataFrame(final_predictions)
output_df.to_excel('test_predictions_3class_new.xlsx', index=False, header=False)

print("Predictions saved to 'test_predictions_3class_new.xlsx'")