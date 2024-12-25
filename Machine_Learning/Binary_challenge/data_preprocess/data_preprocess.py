import os
import cv2
import numpy as np
import pywt
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops, hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


radius = 1
n_points = 8 * radius


def extract_wavelet_features(image):
    resized_image = cv2.resize(image, (128, 128))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
    coeffs2 = pywt.dwt2(gray_image, 'haar')
    LL, (LH, HL, HH) = coeffs2
    wavelet_features = np.hstack([LL.flatten(), LH.flatten(), HL.flatten(), HH.flatten()])
    return wavelet_features


def extract_gabor_features(image):
    resized_image = cv2.resize(image, (128, 128))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
    gabor_kernels = []
    for theta in range(4):  
        theta = theta / 4. * np.pi
        kernel = cv2.getGaborKernel((21, 21), 3.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        gabor_kernels.append(kernel)
    
    gabor_features = []
    for kernel in gabor_kernels:
        fimg = cv2.filter2D(gray_image, cv2.CV_8UC3, kernel)
        gabor_features.append(np.mean(fimg))
        gabor_features.append(np.std(fimg))
    
    return np.array(gabor_features)


def extract_hog_features(image):
    resized_image = cv2.resize(image, (128, 128))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
    hog_features = hog(gray_image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False, multichannel=False)
    return hog_features


def extract_features(image):
    print("Starting feature extraction...")

    
    resized_image = cv2.resize(image, (128, 128))

    
    print("Extracting HSV features...")
    hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2HSV)

    
    h_mean, h_std = np.mean(hsv_image[:, :, 0]), np.std(hsv_image[:, :, 0])
    s_mean, s_std = np.mean(hsv_image[:, :, 1]), np.std(hsv_image[:, :, 1])
    v_mean, v_std = np.mean(hsv_image[:, :, 2]), np.std(hsv_image[:, :, 2])

    
    print("Extracting LBP features...")
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
    lbp_image = local_binary_pattern(gray_image, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2), density=True)

    
    print("Extracting GLCM features...")
    glcm = greycomatrix(gray_image, distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]

    
    print("Extracting Wavelet features...")
    wavelet_features = extract_wavelet_features(image)

    print("Extracting Gabor features...")
    gabor_features = extract_gabor_features(image)

    print("Extracting HOG features...")
    hog_features = extract_hog_features(image)

    
    feature_vector = [
        h_mean, h_std, s_mean, s_std, v_mean, v_std,   
        contrast, energy, homogeneity, correlation     
    ]

    
    feature_vector.extend(lbp_hist)
    feature_vector.extend(wavelet_features)
    feature_vector.extend(gabor_features)
    feature_vector.extend(hog_features)

    print("Feature extraction complete.")
    
    return feature_vector


def process_image_directory(directory, label):
    feature_list = []
    labels = []

    
    for subdir, dirs, files in os.walk(directory):
        for i, file in enumerate(files):
            file_path = os.path.join(subdir, file)
            image = cv2.imread(file_path)
            if image is not None:
                print(f"Processing {file} ({i + 1}/{len(files)}) from {subdir}...")

                
                features = extract_features(image)
                feature_list.append(features)
                labels.append(label)
    
    return feature_list, labels


train_nevus_path = 'train/train/nevus'
train_others_path = 'train/train/others'
val_nevus_path = 'val/val/nevus'
val_others_path = 'val/val/others'


train_nevus_features, train_nevus_labels = process_image_directory(train_nevus_path, label=0)
train_others_features, train_others_labels = process_image_directory(train_others_path, label=1)
val_nevus_features, val_nevus_labels = process_image_directory(val_nevus_path, label=0)
val_others_features, val_others_labels = process_image_directory(val_others_path, label=1)


X_train = np.array(train_nevus_features + train_others_features)
y_train = np.array(train_nevus_labels + train_others_labels)
X_val = np.array(val_nevus_features + val_others_features)
y_val = np.array(val_nevus_labels + val_others_labels)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


np.save('X_train_no_segmentation.npy', X_train)  
np.save('y_train_no_segmentation.npy', y_train)  
np.save('X_val_no_segmentation.npy', X_val)      
np.save('y_val_no_segmentation.npy', y_val)