import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib


X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')

X_train_1 = np.load('X_train_no_segmentation_hair.npy')
y_train_1 = np.load('y_train_no_segmentation_hair.npy')
X_val_1 = np.load('X_val_no_segmentation_hair.npy')
y_val_1 = np.load('y_val_no_segmentation_hair.npy')


X_train_1, y_train_1 = resample(X_train_1, y_train_1, n_samples=len(y_train), random_state=42)


X_val_1, y_val_1 = resample(X_val_1, y_val_1, n_samples=len(y_val), random_state=42)


print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_val:", X_val.shape)
print("Shape of y_val:", y_val.shape)
print("Shape of X_train_1:", X_train_1.shape)
print("Shape of y_train_1:", y_train_1.shape)
print("Shape of X_val_1:", X_val_1.shape)
print("Shape of y_val_1:", y_val_1.shape)


xgb_clf_1 = xgb.XGBClassifier(objective='binary:logistic', n_estimators=400, use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_clf_2 = xgb.XGBClassifier(objective='binary:logistic', n_estimators=400, use_label_encoder=False, eval_metric='logloss', random_state=42)


xgb_clf_1.fit(X_train, y_train)
xgb_clf_2.fit(X_train_1, y_train_1)


train_meta_features1 = xgb_clf_1.predict_proba(X_train)[:, 1]
train_meta_features2 = xgb_clf_2.predict_proba(X_train_1)[:, 1]


X_train_meta = np.column_stack((train_meta_features1, train_meta_features2))


val_meta_features1 = xgb_clf_1.predict_proba(X_val)[:, 1]
val_meta_features2 = xgb_clf_2.predict_proba(X_val_1)[:, 1]


X_val_meta = np.column_stack((val_meta_features1, val_meta_features2))


meta_clf = LogisticRegression(random_state=42)
meta_clf.fit(X_train_meta, y_train)


y_val_pred_meta = meta_clf.predict(X_val_meta)


val_accuracy = accuracy_score(y_val, y_val_pred_meta)
print(f"\nValidation Accuracy (Stacked Model): {val_accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred_meta))
print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_val_pred_meta))

joblib.dump(xgb_clf_1, 'xgb_clf1_model.pkl')
joblib.dump(xgb_clf_2, 'xgb_clf2_model.pkl')
joblib.dump(meta_clf, 'meta_model.pkl')