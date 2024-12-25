
# **Skin Lesion CAD: Machine Learning and Deep Learning Approaches**

## **Overview**
This repository implements both machine learning and deep learning pipelines for the classification of skin lesions, focusing on binary and three-class challenges. The project explores preprocessing techniques, segmentation, hair removal, and ensemble learning methods to achieve robust and interpretable results.

---

## **Repository Structure**
```
CAD/
├── CAD_Presentation_MAIA.pptx              # Project presentation file
├── grab_cut.py                             # Script for Grab Cut segmentation
├── hair_removal.py                         # Script for hair removal preprocessing
├── test_predictions.xlsx                   # Predictions for binary classification
├── test_predictions_3class_new.xlsx        # Predictions for three-class classification
├── Deep_Learning/                          # Deep learning implementation
│   └── Skin_Lesion_CAD.ipynb               # Jupyter Notebook for deep learning approach
├── Machine_Learning/                       # Machine learning implementations
│   ├── Binary_challenge/                   # Binary classification files
│   │   ├── data_preprocess/                # Preprocessing scripts
│   │   │   ├── data_preprocess.py          # General preprocessing
│   │   │   ├── data_preprocess_hair.py     # Hair removal preprocessing
│   │   │   ├── data_preprocess_segmentation.py # Segmentation preprocessing
│   │   │   ├── data_preprocess_segmentation_test.py # Test-time segmentation
│   │   │   └── preprocess_hair_test.py     # Test-time hair removal
│   │   ├── data_test/                      # Testing scripts
│   │   │   └── data_test_binary.py         # Binary classification testing
│   │   ├── data_train/                     # Training scripts
│   │   │   ├── data_train.py               # General training script
│   │   │   ├── data_train_gradient.py      # Gradient-based training
│   │   │   ├── data_train_stacked.py       # Stacked ensemble training
│   │   │   └── data_train_SVM.py           # SVM-specific training
│   ├── Three_Class_Challenge/              # Three-class classification files
│   │   ├── data_preprocess/                # Preprocessing scripts
│   │   │   ├── data_preprocess_3class.py   # General preprocessing
│   │   │   ├── data_preprocess_3class_test.py # Test-time preprocessing
│   │   │   ├── data_preprocess_segmentation_3.py # Segmentation preprocessing
│   │   │   └── data_preprocess_segmentation_3_test.py # Test-time segmentation
```

---

## **File Descriptions**

### **Top-Level Files**
1. **CAD_Presentation_MAIA.pptx**:
   - Presentation summarizing the project results, metrics, and findings.

2. **grab_cut.py**:
   - Implements the Grab Cut algorithm for segmenting skin lesions from dermoscopic images.
   - **Usage**:
     ```bash
     python grab_cut.py
     ```

3. **hair_removal.py**:
   - Applies the Dull Razor Technique to remove hair artifacts from dermoscopic images.
   - **Usage**:
     ```bash
     python hair_removal.py
     ```

4. **test_predictions.xlsx**:
   - Stores predictions for binary classification tasks.

5. **test_predictions_3class_new.xlsx**:
   - Stores predictions for three-class classification tasks.

---

### **Dataset Description**
- **Source**: The dataset contains dermoscopic images for skin lesion analysis, including common types such as melanoma, basal cell carcinoma (BCC), and squamous cell carcinoma (SCC).
- **Classes**:
  - **Binary Classification**:
    - Nevus vs. Others (Dermatofibroma, Melanoma, BCC, SCC).
  - **Three-Class Classification**:
    - Melanoma, BCC, SCC.
- **Structure**:
  - Training and validation datasets are structured into directories by class.
  - Balanced across classes for binary classification; oversampled for SCC in three-class classification.
- **Preprocessing**:
  - Hair removal (Dull Razor Technique).
  - Segmentation (Grab Cut Algorithm).
  - Normalized with mean = `(0.485, 0.456, 0.406)` and std = `(0.229, 0.224, 0.225)`.

---

### **Deep Learning**

#### **Skin_Lesion_CAD.ipynb**
- Implements the deep learning approach using:
  - **Swin Transformer**: Hierarchical transformer for capturing global dependencies.
  - **RegNetY**: A scalable CNN for efficient feature extraction.
  - **Ensemble**: Combines Swin and RegNetY outputs for robust classification.
- **Data Augmentations**:
  - Brightness/contrast adjustments, rotations, flips, affine transformations, coarse dropout.
- **Metrics**:
  - Binary Classification: Accuracy = **93.42%**.
  - Three-Class Classification: Accuracy = **96.93%**, Kappa = **0.9447**.
- **Usage**:
  - Run the notebook step-by-step to preprocess data, train models, and evaluate results.

---

### **Machine Learning**

#### **Binary Challenge**

1. **Preprocessing Scripts**:
   - **data_preprocess.py**: General preprocessing pipeline.
   - **data_preprocess_hair.py**: Incorporates hair removal preprocessing.
   - **data_preprocess_segmentation.py**: Adds segmentation using Grab Cut.
   - **data_preprocess_segmentation_test.py**: Test-time segmentation preprocessing.
   - **preprocess_hair_test.py**: Applies hair removal during testing.

2. **Testing**:
   - **data_test_binary.py**: Tests binary classification models on prepared datasets.

3. **Training**:
   - **data_train.py**: Trains baseline machine learning models.
   - **data_train_gradient.py**: Implements gradient boosting training.
   - **data_train_stacked.py**: Trains stacked ensemble models.
   - **data_train_SVM.py**: Trains SVM for binary classification.

#### **Three-Class Challenge**

1. **Preprocessing Scripts**:
   - **data_preprocess_3class.py**: General preprocessing pipeline.
   - **data_preprocess_3class_test.py**: Test-time preprocessing.
   - **data_preprocess_segmentation_3.py**: Adds segmentation preprocessing.
   - **data_preprocess_segmentation_3_test.py**: Test-time segmentation preprocessing.

---

## **How to Access and Use**

### **Machine Learning**
1. **Preprocessing**:
   - For binary classification:
     ```bash
     python Machine_Learning/Binary_challenge/data_preprocess/data_preprocess.py
     ```
   - For three-class classification:
     ```bash
     python Machine_Learning/Three_Class_Challenge/data_preprocess/data_preprocess_3class.py
     ```

2. **Training**:
   - Train stacked ensemble for binary classification:
     ```bash
     python Machine_Learning/Binary_challenge/data_train/data_train_stacked.py
     ```
   - Train SVM for three-class classification:
     ```bash
     python Machine_Learning/Three_Class_Challenge/data_train/data_train_SVM.py
     ```

3. **Testing**:
   - Test binary classification models:
     ```bash
     python Machine_Learning/Binary_challenge/data_test/data_test_binary.py
     ```

### **Deep Learning**
1. Open `Deep_Learning/Skin_Lesion_CAD.ipynb` in Jupyter Notebook or Google Colab.
2. Follow the step-by-step cells to:
   - Preprocess the data.
   - Train Swin Transformer and RegNetY models.
   - Evaluate the ensemble model.

---

## **Evaluation and Results**

### **Machine Learning**
- **Binary Classification**:
  - Stacked Ensemble: Accuracy = ~82.83%.
- **Three-Class Classification**:
  - Gradient Boosting: Accuracy = ~78.28%.

### **Deep Learning**
- **Binary Classification**:
  - Swin + RegNetY Ensemble: Accuracy = **93.42%**.
- **Three-Class Classification**:
  - Swin + RegNetY Ensemble: Accuracy = **96.93%**, Kappa = **0.9447**.

---

## **Contact**
For questions or feedback, feel free to raise an issue or contact the repository owner.
