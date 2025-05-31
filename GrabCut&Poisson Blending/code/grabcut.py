import numpy as np
import cv2
import argparse
from utils import build_graph, vertex_to_pixel  
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import igraph as ig

GC_BGD = 0 # Hard bg pixel
GC_FGD = 1 # Hard fg pixel, will not be used
GC_PR_BGD = 2 # Soft bg pixel
GC_PR_FGD = 3 # Soft fg pixel

first_iter = True

# Function to resize mask to target shape
def resize_mask(mask, target_shape, interpolation=cv2.INTER_NEAREST):
    return cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=interpolation)

# Main GrabCut function
def grabcut(img, rect, scale_factor=0.5, n_iter=1000):
    original_shape = img.shape[:2]

    # Apply various blurs to the image
    img = cv2.medianBlur(img, 5)

    
    small_img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    small_rect = (int(rect[0] * scale_factor), int(rect[1] * scale_factor), int(rect[2] * scale_factor), int(rect[3] * scale_factor))

    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(small_img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = small_rect
    
    w -= x
    h -= y

    # Initialize the inner square to Foreground
    mask[y:y + h, x:x + w] = GC_PR_FGD
    mask[small_rect[1] + small_rect[3] // 2, small_rect[0] + small_rect[2] // 2] = GC_FGD

    bgGMM, fgGMM = initialize_GMMs(small_img, mask)
    prev_bgd_indices, prev_fgd_indices = -1, -1  # Initialize previous pixel counts for convergence check

    for i in range(n_iter):
        # Update GMM
        bgGMM, fgGMM = update_GMMs(small_img, mask, bgGMM, fgGMM)

        mincut_sets, energy = calculate_mincut(small_img, mask, bgGMM, fgGMM)

        mask = update_mask(mincut_sets, mask)

        bgd_indices = np.where((mask == GC_BGD) | (mask == GC_PR_BGD))[0].size
        fgd_indices = np.where((mask == GC_FGD) | (mask == GC_PR_FGD))[0].size
        
        # Check for convergence based on pixel count change
        if check_convergence(bgd_indices, fgd_indices, prev_bgd_indices, prev_fgd_indices, energy):
            break

        prev_bgd_indices, prev_fgd_indices = bgd_indices, fgd_indices

    final_mask = resize_mask(mask, original_shape)
    
    # Return the final mask and the GMMs
    final_mask = np.where((final_mask == GC_FGD) | (final_mask == GC_PR_FGD), 1, 0).astype(np.uint8)

    return final_mask, bgGMM, fgGMM

# Function to initialize GMMs
def initialize_GMMs(img, mask, n_components=5):
    """
    Initialize the GMMs for foreground and background using clustering and GaussianMixture.

    Parameters:
    - img: Input image
    - mask: Initial mask (with unknown, foreground, and background labels)
    - n_components: Number of components in the GMM

    Returns:
    - bgdGMM: Initialized GMM for the background
    - fgdGMM: Initialized GMM for the foreground
    """
    # Get indices of background and foreground pixels based on the mask
    bgd_indices = np.where((mask == GC_BGD) | (mask == GC_PR_BGD))
    fgd_indices = np.where((mask == GC_FGD) | (mask == GC_PR_FGD))

    # Extract the background and foreground pixels
    bgd_pixels = img[bgd_indices].reshape(-1, 3)
    fgd_pixels = img[fgd_indices].reshape(-1, 3)

    # Ensure we have at least 1 component
    n_components_bg = max(min(n_components, bgd_pixels.shape[0]), 1)
    n_components_fg = max(min(n_components, fgd_pixels.shape[0]), 1)

    # Use KMeans to cluster the pixels into K components
    kmeans_bgd = KMeans(n_clusters=n_components_bg, init='k-means++').fit(bgd_pixels)
    kmeans_fgd = KMeans(n_clusters=n_components_fg, init='k-means++').fit(fgd_pixels)

    # Initialize Gaussian Mixture Models for background and foreground
    bgdGMM = GaussianMixture(n_components=n_components_bg, covariance_type='full')
    fgdGMM = GaussianMixture(n_components=n_components_fg, covariance_type='full')

    # Set initial means of GMMs to cluster centers from KMeans
    bgdGMM.means_init = kmeans_bgd.cluster_centers_
    fgdGMM.means_init = kmeans_fgd.cluster_centers_

    # Set initial weights and covariances for GMMs
    bgdGMM.weights_init = np.array([np.sum(kmeans_bgd.labels_ == i) for i in range(n_components_bg)]) / len(bgd_pixels)
    fgdGMM.weights_init = np.array([np.sum(kmeans_fgd.labels_ == i) for i in range(n_components_fg)]) / len(fgd_pixels)
    bgdGMM.covariances_init = np.array([np.cov(bgd_pixels[kmeans_bgd.labels_ == i].T) for i in range(n_components_bg)])
    fgdGMM.covariances_init = np.array([np.cov(fgd_pixels[kmeans_fgd.labels_ == i].T) for i in range(n_components_fg)])

    # Ensure covariances are positive semi-definite
    for cov in bgdGMM.covariances_init:
        if np.linalg.det(cov) <= 0:
            cov += np.eye(cov.shape[0]) * 1e-6
    for cov in fgdGMM.covariances_init:
        if np.linalg.det(cov) <= 0:
            cov += np.eye(cov.shape[0]) * 1e-6

    # Compute Cholesky decomposition of the covariances for the GMMs
    bgdGMM.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(bgdGMM.covariances_init))
    fgdGMM.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(fgdGMM.covariances_init))

    # Fit the GMMs to the clustered data
    bgdGMM.fit(bgd_pixels)
    fgdGMM.fit(fgd_pixels)

    return bgdGMM, fgdGMM

# Function to update GMMs
def update_GMMs(img, mask, bgdGMM, fgdGMM, n_components=5, epsilon=1e-5):
    """
    Update the GMMs for background and foreground based on current segmentation.

    Parameters:
    - img: Input image
    - mask: Current mask
    - bgdGMM: Background GMM
    - fgdGMM: Foreground GMM
    - n_components: Number of components in the GMM
    - epsilon: Small value to ensure positive semi-definite covariance matrices

    Returns:
    - Updated bgdGMM and fgdGMM
    """
    # Get indices of background and foreground pixels based on the mask
    bgd_indices = np.where((mask == GC_BGD) | (mask == GC_PR_BGD))
    fgd_indices = np.where((mask == GC_FGD) | (mask == GC_PR_FGD))

    # Extract the background and foreground pixels
    bgd_pixels = img[bgd_indices].reshape(-1, 3)
    fgd_pixels = img[fgd_indices].reshape(-1, 3)

    # Check if there are enough pixels to update the GMMs
    if len(bgd_pixels) < n_components or len(fgd_pixels) < n_components:
        return bgdGMM, fgdGMM

    # Predict labels for the pixels using the GMMs
    bgd_labels = bgdGMM.predict(bgd_pixels)
    fgd_labels = fgdGMM.predict(fgd_pixels)

    for k in range(n_components):
        # Update background GMM parameters
        bgd_cluster_pixels = bgd_pixels[bgd_labels == k]
        if len(bgd_cluster_pixels) > 0:
            bgdGMM.means_[k] = bgd_cluster_pixels.mean(axis=0)
            bgdGMM.covariances_[k] = np.cov(bgd_cluster_pixels.T) + epsilon * np.eye(3)
            bgdGMM.weights_[k] = len(bgd_cluster_pixels) / len(bgd_pixels)
        else:
            bgdGMM.covariances_[k] = epsilon * np.eye(3)

        # Update foreground GMM parameters
        fgd_cluster_pixels = fgd_pixels[fgd_labels == k]
        if len(fgd_cluster_pixels) > 0:
            fgdGMM.means_[k] = fgd_cluster_pixels.mean(axis=0)
            fgdGMM.covariances_[k] = np.cov(fgd_cluster_pixels.T) + epsilon * np.eye(3)
            fgdGMM.weights_[k] = len(fgd_cluster_pixels) / len(fgd_pixels)
        else:
            fgdGMM.covariances_[k] = epsilon * np.eye(3)

        # Ensure covariances are positive semi-definite
        if not np.all(np.isfinite(bgdGMM.covariances_[k])) or not np.all(np.linalg.eigvals(bgdGMM.covariances_[k]) > 0):
            bgdGMM.covariances_[k] = epsilon * np.eye(3)
        if not np.all(np.isfinite(fgdGMM.covariances_[k])) or not np.all(np.linalg.eigvals(fgdGMM.covariances_[k]) > 0):
            fgdGMM.covariances_[k] = epsilon * np.eye(3)

    # Compute Cholesky decomposition of the updated covariances for the GMMs
    bgdGMM.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(bgdGMM.covariances_))
    fgdGMM.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(fgdGMM.covariances_))

    return bgdGMM, fgdGMM

# Function to calculate mincut
def calculate_mincut(img, mask, bgGMM, fgGMM):
    graph = build_graph(img, mask, bgGMM, fgGMM)
    h, w, _ = img.shape
    num_pixels = h * w
    source = num_pixels
    sink = num_pixels + 1
    mincut = graph.st_mincut(source, sink, capacity='capacity')
    return mincut, mincut.value

# Function to update mask
def update_mask(mincut_sets, mask):
    h, w = mask.shape
    new_mask = np.copy(mask)

    source_set = set(mincut_sets[0])
    sink_set = set(mincut_sets[1])

    updated_source_pixels = 0
    updated_sink_pixels = 0

    source = h * w  # Source node
    sink = h * w + 1  # Sink node

    for node in source_set:
        if node == source or node == sink:
            continue
        i, j = vertex_to_pixel(node, w)
        if 0 <= i < h and 0 <= j < w and mask[i, j] == GC_PR_BGD:
            new_mask[i, j] = GC_PR_FGD
            updated_source_pixels += 1

    for node in sink_set:
        if node == source or node == sink:
            continue
        i, j = vertex_to_pixel(node, w)
        if 0 <= i < h and 0 <= j < w and (mask[i, j] == GC_PR_FGD or mask[i,j] == GC_FGD):
            new_mask[i, j] = GC_PR_BGD
            updated_sink_pixels += 1

    return new_mask

# Function to check convergence
prev_energy = None

def check_convergence(bgd_indices, fgd_indices, prev_bgd_indices, prev_fgd_indices, energy, threshold=1e-4):
    global prev_energy
    if prev_energy is None:
        prev_energy = energy
        return False
    diff = abs(prev_energy - energy)
    prev_energy = energy
    return diff < threshold or (abs(bgd_indices - prev_bgd_indices) < 1 and abs(fgd_indices - prev_fgd_indices) < 10)

# Function to calculate metrics
def cal_metric(predicted_mask, gt_mask):
    intersection = np.logical_and(predicted_mask == GC_FGD, gt_mask == GC_FGD).sum()
    union = np.logical_or(predicted_mask == GC_FGD, gt_mask == GC_FGD).sum()
    
    iou = intersection / union if union != 0 else 0
    accuracy = (predicted_mask == gt_mask).sum() / gt_mask.size
    
    return accuracy * 100, iou * 100

# Argument parser
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='cross', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()

if __name__ == '__main__':
    # Load an example image and define a bounding box around the object of interest
    args = parse()

    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int, args.rect.split(',')))

    img = cv2.imread(input_path)

    # Run the GrabCut algorithm on the image and bounding box
    mask, bgGMM, fgGMM = grabcut(img, rect)
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = cal_metric(mask, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])
   # Optionally display the results
    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCut Mask', 255 * mask)
    cv2.imshow('GrabCut Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
