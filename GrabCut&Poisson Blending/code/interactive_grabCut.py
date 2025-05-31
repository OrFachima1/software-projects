import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from grabcut import grabcut, cal_metric  # ייבוא הפונקציות מהקובץ grabcut.py

def list_images_in_directory(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def select_image(images):
    print("Available images:")
    for idx, img in enumerate(images):
        print(f"{idx + 1}. {img}")
    
    choice = int(input("Select an image by number: "))
    return images[choice - 1]

def select_rectangle(img):
    r = cv2.selectROI("Select ROI", img, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")
    return (int(r[0]), int(r[1]), int(r[2]), int(r[3]))

def main():
    images_dir = "data/imgs"
    masks_dir = "data/seg_GT"
    bboxes_dir = "data/bboxes"

    images = list_images_in_directory(images_dir)
    if not images:
        print("No images found in the directory.")
        return

    selected_image = select_image(images)
    img_path = os.path.join(images_dir, selected_image)
    img = cv2.imread(img_path)

    use_predefined_mask = input("Do you want to use a predefined mask? (y/n): ").strip().lower() == 'y'

    if use_predefined_mask:
        bbox_path = os.path.join(bboxes_dir, selected_image.replace('.jpg', '.txt'))
        with open(bbox_path, 'r') as f:
            rect = tuple(map(int, f.read().strip().split()))
        gt_mask_path = os.path.join(masks_dir, selected_image.replace('.jpg', '.bmp'))
        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
    else:
        rect = select_rectangle(img)
        gt_mask = None

    mask, bgGMM, fgGMM = grabcut(img, rect)

    if gt_mask is not None:
        acc, jac = cal_metric(mask, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')

     # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])
    cv2.imwrite('result.bmp', 255 * mask)

    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCut Mask', 255 * mask)
    cv2.imshow('GrabCut Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
