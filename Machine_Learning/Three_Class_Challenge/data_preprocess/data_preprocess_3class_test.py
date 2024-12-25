import os
import cv2
import numpy as np
import pywt
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops, hog
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


radius = 1
n_points = 8 * radius


lock = Lock()
progress_counter = 0


def remove_hair(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    bhg = cv2.GaussianBlur(blackhat, (3, 3), cv2.BORDER_DEFAULT)
    _, mask = cv2.threshold(bhg, 10, 255, cv2.THRESH_BINARY)
    cleaned_image = cv2.inpaint(image, mask, 6, cv2.INPAINT_TELEA)
    return cleaned_image


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
    cleaned_image = remove_hair(image)
    resized_image = cv2.resize(cleaned_image, (128, 128))
    hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2HSV)

    
    h_mean, h_std = np.mean(hsv_image[:, :, 0]), np.std(hsv_image[:, :, 0])
    s_mean, s_std = np.mean(hsv_image[:, :, 1]), np.std(hsv_image[:, :, 1])
    v_mean, v_std = np.mean(hsv_image[:, :, 2]), np.std(hsv_image[:, :, 2])

    
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
    lbp_image = local_binary_pattern(gray_image, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2), density=True)

    
    glcm = greycomatrix(gray_image, distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]

    
    wavelet_features = extract_wavelet_features(cleaned_image)
    gabor_features = extract_gabor_features(cleaned_image)
    hog_features = extract_hog_features(cleaned_image)

    
    feature_vector = [
        h_mean, h_std, s_mean, s_std, v_mean, v_std,  
        contrast, energy, homogeneity, correlation    
    ]
    feature_vector.extend(lbp_hist)
    feature_vector.extend(wavelet_features)
    feature_vector.extend(gabor_features)
    feature_vector.extend(hog_features)

    return feature_vector


def process_single_image(file_path, total_files):
    global progress_counter
    image = cv2.imread(file_path)
    if image is not None:
        features = extract_features(image)
        
        with lock:
            progress_counter += 1
            print(f"Processed {progress_counter}/{total_files} images")
        return features
    return None


def process_image_directory(directory):
    feature_list = []
    file_paths = [os.path.join(subdir, file) for subdir, dirs, files in os.walk(directory) for file in files]
    total_files = len(file_paths)

    
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_single_image, file, total_files): file for file in file_paths}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                feature_list.append(result)
    
    return feature_list


test_image_path = 'test_3/test_3'


X_test_features = process_image_directory(test_image_path)


X_test_features = np.array(X_test_features)
np.save('X_test_3.npy', X_test_features)

print("Test image feature extraction and hair removal complete. Data saved to 'X_test.npy'.")