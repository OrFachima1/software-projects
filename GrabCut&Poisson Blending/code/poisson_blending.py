import cv2
import numpy as np
import scipy.sparse
from scipy.sparse.linalg import spsolve
import argparse

def poisson_blend(im_src, im_tgt, im_mask, center):
    # Convert images to float32 to avoid overflow during computation
    im_src = im_src.astype(np.float32)
    im_tgt = im_tgt.astype(np.float32)

    # Get the mask indices
    mask_indices = np.where(im_mask != 0)
    num_pixels = len(mask_indices[0])

    # Get the shape of the source and target images
    h_src, w_src = im_src.shape[:2]
    h_tgt, w_tgt = im_tgt.shape[:2]

    # Create a sparse matrix A and the vector b
    A = scipy.sparse.lil_matrix((num_pixels, num_pixels))
    b = np.zeros((num_pixels, 3), dtype=np.float32)

    # Compute the Laplacian operator
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)

    # Create a dictionary to map (y, x) to index in A and b
    index_map = {(y, x): i for i, (y, x) in enumerate(zip(mask_indices[0], mask_indices[1]))}

    # Populate the sparse matrix A and the vector b
    for i, (y, x) in enumerate(zip(mask_indices[0], mask_indices[1])):
        A[i, i] = 4
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h_src and 0 <= nx < w_src:
                if im_mask[ny, nx] != 0:
                    index = index_map[(ny, nx)]
                    A[i, index] = -1
                else:
                    tgt_y = ny + center[1] - h_src // 2
                    tgt_x = nx + center[0] - w_src // 2
                    if 0 <= tgt_y < h_tgt and 0 <= tgt_x < w_tgt:
                        b[i] += im_tgt[tgt_y, tgt_x]
            else:
                tgt_y = ny + center[1] - h_src // 2
                tgt_x = nx + center[0] - w_src // 2
                if 0 <= tgt_y < h_tgt and 0 <= tgt_x < w_tgt:
                    b[i] += im_tgt[tgt_y, tgt_x]

        # Compute the Laplacian contribution from the source image
        for c in range(3):
            if 0 <= y-1 < h_src and 0 <= y+1 < h_src and 0 <= x-1 < w_src and 0 <= x+1 < w_src:
                b[i, c] += np.sum(laplacian * im_src[y-1:y+2, x-1:x+2, c])

    # Convert A to CSR format for faster computation
    A = A.tocsr()

    # Solve the Poisson equation for each color channel separately
    blended_channels = np.zeros((num_pixels, 3), dtype=np.float32)
    for c in range(3):
        blended_channels[:, c] = spsolve(A, b[:, c])

    # Create an empty image to hold the blended result
    im_blended_result = np.zeros_like(im_tgt, dtype=np.float32)

    # Set the solution to the masked area in the blended result image
    for i, (y, x) in enumerate(zip(mask_indices[0], mask_indices[1])):
        tgt_y = y + center[1] - h_src // 2
        tgt_x = x + center[0] - w_src // 2
        if 0 <= tgt_y < h_tgt and 0 <= tgt_x < w_tgt:
            im_blended_result[tgt_y, tgt_x] = blended_channels[i]

    # Clip the result to the valid range [0, 255] and convert to uint8
    im_blended_result_display = np.clip(im_blended_result, 0, 255).astype(np.uint8)

    # Copy the target image before updating
    im_blend = im_tgt.copy()

    # Update only the mask region in the target image based on the Poisson result
    for i, (y, x) in enumerate(zip(mask_indices[0], mask_indices[1])):
        tgt_y = y + center[1] - h_src // 2
        tgt_x = x + center[0] - w_src // 2
        if 0 <= tgt_y < h_tgt and 0 <= tgt_x < w_tgt:
            im_blend[tgt_y, tgt_x] = im_blended_result_display[tgt_y, tgt_x]

    # Clip the final result to the valid range [0, 255] and convert to uint8
    im_blend_display = np.clip(im_blend, 0, 255).astype(np.uint8)

    return im_blend_display

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/soval.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/result.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/IMG_4701.jpg', help='mask file path')
    return parser.parse_args()


if __name__ == "__main__":
    # Load the source and target images
    args = parse()

    im_tgt = cv2.imread(args.tgt_path, cv2.IMREAD_COLOR)
    im_src = cv2.imread(args.src_path, cv2.IMREAD_COLOR)
    if args.mask_path == '':
        im_mask = np.full(im_src.shape[:2], 255, dtype=np.uint8)
    else:
        im_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]

    center = (int(im_tgt.shape[1] / 2), int(im_tgt.shape[0] / 2))

    # Ensure the object within the mask fits within the target image
    mask_indices = np.where(im_mask != 0)
    min_y, max_y = np.min(mask_indices[0]), np.max(mask_indices[0])
    min_x, max_x = np.min(mask_indices[1]), np.max(mask_indices[1])
    mask_height, mask_width = max_y - min_y + 1, max_x - min_x + 1
    h_src, w_src = im_src.shape[:2]
    h_tgt, w_tgt = im_tgt.shape[:2]
    if mask_height > h_tgt or mask_width > w_tgt:
        raise ValueError("The object within the mask must fit within the target image.")

    im_clone = poisson_blend(im_src, im_tgt, im_mask, center)

    cv2.imwrite('fullmoon_result.png', im_clone)
    cv2.imshow('Cloned image', im_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
