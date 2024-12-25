import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np


X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')

X_train_1 = np.load('X_train_no_segmentation_hair.npy')
y_train_1 = np.load('y_train_no_segmentation_hair.npy')
X_val_1 = np.load('X_val_no_segmentation_hair.npy')
y_val_1 = np.load('y_val_no_segmentation_hair.npy')



print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_val:", X_val.shape)
print("Shape of y_val:", y_val.shape)

print("Shape of X_train_1:", X_train_1.shape)
print("Shape of y_train_1:", y_train_1.shape)
print("Shape of X_val_1:", X_val_1.shape)
print("Shape of y_val_1:", y_val_1.shape)



xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',   
    n_estimators=400              
   
)


xgb_clf.fit(X_train, y_train)


y_val_pred = xgb_clf.predict(X_val)


val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"\nValidation Accuracy: {val_accuracy * 100:.2f}%")


print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))