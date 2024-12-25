from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np


X_train = np.load('X_train_no_segmentation_hair.npy')
y_train = np.load('y_train_no_segmentation_hair.npy')
X_val = np.load('X_val_no_segmentation_hair.npy')
y_val = np.load('y_val_no_segmentation_hair.npy')


rf = RandomForestClassifier(random_state=42)


rf.fit(X_train, y_train)


y_val_pred = rf.predict(X_val)


val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"\nValidation Accuracy: {val_accuracy * 100:.2f}%")


print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))