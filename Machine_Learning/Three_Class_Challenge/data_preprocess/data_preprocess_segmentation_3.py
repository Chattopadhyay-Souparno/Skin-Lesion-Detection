import os
import cv2
import numpy as np
import pywt
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog
from sklearn.cluster import KMeans
import random
from multiprocessing import Pool, cpu_count
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


radius = 1
n_points = 8 * radius


def kmeans_segmentation(image, K=8):
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    kmeans_img = center[label.flatten()]
    kmeans_img = kmeans_img.reshape(image.shape)
    return kmeans_img


def adaptive_hist_equalization(image):
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))
    h, s, v = cv2.split(image)
    h_eq = clahe.apply(h)
    s_eq = clahe.apply(s)
    v_eq = clahe.apply(v)
    hsv_eq = cv2.merge((h_eq, s_eq, v_eq))
    return hsv_eq


def create_green_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([50, 100, 100])
    upper_green = np.array([100, 255, 255])
    mask_g = cv2.inRange(hsv, lower_green, upper_green)
    return mask_g


def grabcut_with_mask(image, mask):
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    
    mask_gc = np.zeros(image.shape[:2], np.uint8)
    mask_gc[mask == 0] = 0  
    mask_gc[mask == 255] = 1  
    
    
    if not (np.any(mask_gc == 0) and np.any(mask_gc == 1)):
        print("Not enough background/foreground in mask, switching to rectangle GrabCut.")
        return grabcut_with_rectangle(image)

    
    cv2.grabCut(image, mask_gc, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    final_mask = np.where((mask_gc == 2) | (mask_gc == 0), 0, 1).astype('uint8')
    return final_mask


def grabcut_with_rectangle(image):
    height, width = image.shape[:2]
    rect = (int(0.03 * width), int(0.03 * height), int(0.94 * width), int(0.94 * height))
    mask_gc = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask_gc, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    final_mask = np.where((mask_gc == 2) | (mask_gc == 0), 0, 1).astype('uint8')
    return final_mask


def extract_features_from_image(image, is_segmented=False):
    
    resized_image = cv2.resize(image, (128, 128))

    
    hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
    h_mean, h_std = np.mean(hsv_image[:, :, 0]), np.std(hsv_image[:, :, 0])
    s_mean, s_std = np.mean(hsv_image[:, :, 1]), np.std(hsv_image[:, :, 1])
    v_mean, v_std = np.mean(hsv_image[:, :, 2]), np.std(hsv_image[:, :, 2])

    
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1.5)

    
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    sobel_mean, sobel_std = np.mean(sobel_combined), np.std(sobel_combined)

    
    lbp_image = local_binary_pattern(blurred_image, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2), density=True)

    
    glcm = graycomatrix(blurred_image, distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    
    coeffs2 = pywt.dwt2(blurred_image, 'haar')
    LL, (LH, HL, HH) = coeffs2
    wavelet_features = np.hstack([LL.flatten(), LH.flatten(), HL.flatten(), HH.flatten()])

    
    gabor_kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        kernel = cv2.getGaborKernel((21, 21), 3.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        gabor_kernels.append(kernel)

    gabor_features = []
    for kernel in gabor_kernels:
        fimg = cv2.filter2D(blurred_image, cv2.CV_8UC3, kernel)
        gabor_features.append(np.mean(fimg))
        gabor_features.append(np.std(fimg))

    
    hog_features = hog(blurred_image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False, channel_axis=None)

    
    feature_vector = [
        h_mean, h_std, s_mean, s_std, v_mean, v_std,  
        sobel_mean, sobel_std,                       
        contrast, energy, homogeneity, correlation    
    ]
    feature_vector.extend(lbp_hist)
    feature_vector.extend(wavelet_features)
    feature_vector.extend(gabor_features)
    feature_vector.extend(hog_features)

    return feature_vector

def process_single_image(file_path, label, index=None, total=None):
    image = cv2.imread(file_path)
    if image is not None:
        
        features_full_image = extract_features_from_image(image)

        
        kmeans_img = kmeans_segmentation(image, K=8)
        hsv_image = cv2.cvtColor(kmeans_img, cv2.COLOR_BGR2HSV)
        hsv_eq = adaptive_hist_equalization(hsv_image)
        enhanced_image = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
        green_mask = create_green_mask(enhanced_image)
        inv_mask = cv2.bitwise_not(green_mask)

        
        if np.sum(inv_mask) < 80039400:
            grabcut_mask = grabcut_with_mask(image, inv_mask)
        else:
            grabcut_mask = grabcut_with_rectangle(image)

        grabcut_img = image * grabcut_mask[:, :, np.newaxis]

        
        features_segmented_image = extract_features_from_image(grabcut_img, is_segmented=True)

        
        combined_features = np.concatenate([features_full_image, features_segmented_image])

        
        print(f"Processed image {index}/{total}: {file_path}. {total - index} images left.")
        return (combined_features, label)

    return None


def process_image_directory_sample(directory, label, sample_size=5):
    file_paths = [os.path.join(subdir, file) for subdir, dirs, files in os.walk(directory) for file in files]
    sampled_file_paths = random.sample(file_paths, min(sample_size, len(file_paths)))

    total_files = len(sampled_file_paths)
    process_args = [(file_path, label, i + 1, total_files) for i, file_path in enumerate(sampled_file_paths)]

    
    with Pool(cpu_count()) as p:
        results = p.starmap(process_single_image, process_args)

    results = [r for r in results if r is not None]
    feature_list, labels = zip(*results)
    return list(feature_list), list(labels)

if __name__ == '__main__':

    train_mel_path = 'train_3/train_3/mel'
    train_bcc_path = 'train_3/train_3/bcc'
    train_scc_path = 'train_3/train_3/scc'
    val_mel_path = 'val_3/val_3/mel'
    val_bcc_path = 'val_3/val_3/bcc'
    val_scc_path = 'val_3/val_3/scc'

    
    

    
    train_mel_features, train_mel_labels = process_image_directory_sample(train_mel_path, label=0, sample_size= 2700)
    train_bcc_features, train_bcc_labels = process_image_directory_sample(train_bcc_path, label=1, sample_size= 1990)
    train_scc_features, train_scc_labels = process_image_directory_sample(train_scc_path, label=2, sample_size= 370)

    
    train_scc_features, train_scc_labels = resample(train_scc_features, train_scc_labels,
                                                    replace=True,
                                                    n_samples=int(len(train_bcc_features) * 0.9),
                                                    random_state=42)

    
    X_train = np.array(train_mel_features + train_bcc_features + train_scc_features)
    y_train = np.array(train_mel_labels + train_bcc_labels + train_scc_labels)

    
    val_mel_features, val_mel_labels = process_image_directory_sample(val_mel_path, label=0, sample_size=600)
    val_bcc_features, val_bcc_labels = process_image_directory_sample(val_bcc_path, label=1, sample_size=450)
    val_scc_features, val_scc_labels = process_image_directory_sample(val_scc_path, label=2, sample_size=90)

    X_val = np.array(val_mel_features + val_bcc_features + val_scc_features)
    y_val = np.array(val_mel_labels + val_bcc_labels + val_scc_labels)

    from sklearn.utils import shuffle
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_val, y_val = shuffle(X_val, y_val, random_state=42)

    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    
    
    
    

    
    np.save('X_train_3_segmentation.npy', X_train_scaled)
    np.save('y_train_3_segmentation.npy', y_train)
    np.save('X_val_3_segmentation.npy', X_val_scaled)
    np.save('y_val_3_segmentation.npy', y_val)

    print("Feature extraction complete. Data saved to .npy files.")