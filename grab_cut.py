import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


def visualize_step(title, image, cmap=None):
    plt.figure()
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.show()


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


def binarize_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    return binary_mask


def pipeline(image_path):
    
    image = cv2.imread(image_path)
    visualize_step("Original Image", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    
    kmeans_img = kmeans_segmentation(image, K=8)
    visualize_step("K-means Segmentation", cv2.cvtColor(kmeans_img, cv2.COLOR_BGR2RGB))
    
    
    hsv_image = cv2.cvtColor(kmeans_img, cv2.COLOR_BGR2HSV)
    hsv_eq = adaptive_hist_equalization(hsv_image)
    visualize_step("HSV after Adaptive Histogram Equalization", hsv_eq, cmap='gray')
    
    
    enhanced_image = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
    visualize_step("Enhanced Image (BGR)", cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
    
    
    green_mask = create_green_mask(enhanced_image)
    visualize_step("Green Mask", green_mask, cmap='gray')
    
    
    inv_mask = cv2.bitwise_not(green_mask)
    visualize_step("Inverse Green Mask", inv_mask, cmap='gray')
    
    
    foreground_pixels = np.sum(inv_mask == 255)
    background_pixels = np.sum(inv_mask == 0)
    total_pixels = inv_mask.size
    min_percentage = 3  

    foreground_percentage = (foreground_pixels / total_pixels) * 100
    background_percentage = (background_pixels / total_pixels) * 100

    if foreground_percentage > min_percentage and background_percentage > min_percentage:
        print("Applying mask-based GrabCut")
        grabcut_mask = grabcut_with_mask(image, inv_mask)
    else:
        print("Applying rectangle-based GrabCut")
        grabcut_mask = grabcut_with_rectangle(image)
    
    grabcut_img = image * grabcut_mask[:, :, np.newaxis]
    visualize_step("GrabCut Segmented Image", cv2.cvtColor(grabcut_img, cv2.COLOR_BGR2RGB))
    
    
    binarized_img = binarize_image(grabcut_img)
    visualize_step("Binarized Image", binarized_img, cmap='gray')


image_path = 'train/train/nevus/nev00001.jpg'  
pipeline(image_path)