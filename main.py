import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# ============================================================
# CNN KERNEL LAB
#
# Goal:
# Build a tiny feature extractor for classifying circles vs squares.
#
# For each image, your program should:
#   1. apply several kernels
#   2. apply ReLU
#   3. apply max pooling
#   4. turn each pooled feature map into one score
#   5. use those scores to predict the class
# ============================================================


# ============================================================
# SECTION 1: SETTINGS
#
# In this section, set the folder names and image size.
# Do not change the helper code below unless you need to.
# ============================================================

CLASS_A_FOLDER = "images/circles"
CLASS_B_FOLDER = "images/squares"
IMAGE_SIZE = (128, 128)


# ============================================================
# SECTION 2: LOAD IMAGES
#
# This helper function is already written for you.
# It loads every image in a folder, resizes it, and scales
# pixel values to be between 0 and 1.
# ============================================================

def load_images(folder, size):
    images = []
    filenames = []

    for filename in sorted(os.listdir(folder)):
        path = os.path.join(folder, filename)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, size)
        img = img.astype(np.float32) / 255.0

        images.append(img)
        filenames.append(filename)

    return images, filenames


# ============================================================
# SECTION 3: RELU
#
# TASK:
# Write a function called relu(x).
#
# It should replace all negative values in x with 0.
#
# Example:
#   relu([-2, 3, -1, 5]) should become [0, 3, 0, 5]
# ============================================================

def relu(x):
    return np.maximum(x, 0)




# ============================================================
# SECTION 4: MAX POOLING
#
# TASK:
# Write a function called max_pool(img, size=2).
#
# This function should:
#   - break the image into non-overlapping size x size blocks
#   - replace each block with its maximum value
#
# Example:
#   a 2x2 block like
#       [1 4]
#       [2 3]
#   should become
#       [4]
#
# Hint:
#   You may assume the image is 2D.
#   You may crop off extra rows or columns if needed.
# ============================================================

def max_pool(img, size=2):
    h, w = img.shape
    h2 = h - (h % size)
    w2 = w - (w % size)
    img = img[:h2, :w2]

    return img.reshape(h2 // size, size, w2 // size, size).max(axis=(1, 3))



# ============================================================
# SECTION 5: STUDENT KERNELS
#
# TASK:
# Create THREE different 3x3 kernels.
#
# Suggestions:
#   - one edge detector
#   - one kernel that might respond to corners
#   - one custom kernel of your own design
#
# Store them in a list called kernels.
#
# Example structure:
#   kernel1 = np.array([...], dtype=np.float32)
#   kernel2 = ...
#   kernel3 = ...
#   kernels = [kernel1, kernel2, kernel3]
# ============================================================

# TODO: define kernel1
# TODO: define kernel2
# TODO: define kernel3

kernels = []


# ============================================================
# SECTION 6: APPLY ONE KERNEL TO ONE IMAGE
#
# TASK:
# Complete the function process_image(image, kernel).
#
# It should:
#   1. apply the kernel using cv2.filter2D
#   2. apply ReLU
#   3. apply max pooling
#   4. compute one score from the pooled map
#
# For the score, use the maximum value in the pooled map.
#
# The function should return:
#   feature_map, pooled_map, score
# ============================================================

def process_image(image, kernel):
    # TODO: apply the kernel
    feature_map = None

    # TODO: apply ReLU
    feature_map = None

    # TODO: apply max pooling
    pooled_map = None

    # TODO: compute one score from the pooled map
    score = None

    return feature_map, pooled_map, score


# ============================================================
# SECTION 7: EXTRACT FEATURES FROM ONE IMAGE
#
# TASK:
# Complete the function extract_features(image, kernels).
#
# It should:
#   - apply every kernel in the list
#   - get one score from each kernel
#   - return the list of scores
#
# Example:
#   if there are 3 kernels, the output might look like
#   [2.1, 0.8, 1.7]
# ============================================================

def extract_features(image, kernels):
    features = []

    # TODO: loop through the kernels
    # TODO: process the image with each kernel
    # TODO: add each score to the features list

    return features


# ============================================================
# SECTION 8: MAKE A PREDICTION
#
# TASK:
# Write a simple rule to predict whether an image is a circle
# or a square using the feature list.
#
# Return:
#   0 for circles
#   1 for squares
#
# You must decide on your own rule.
#
# Example ideas:
#   - compare feature 0 and feature 1
#   - add some features together
#   - use np.argmax if your features represent class evidence
# ============================================================

def predict_class(features):
    # TODO: write your classification rule
    prediction = None
    return prediction


# ============================================================
# SECTION 9: EVALUATE MANY IMAGES
#
# TASK:
# Complete this function so that it:
#   - extracts features for each image
#   - predicts the class
#   - checks whether the prediction is correct
#   - counts how many were correct
#
# The function should return the accuracy as a decimal.
#
# true_label will be:
#   0 for circles
#   1 for squares
# ============================================================

def evaluate_dataset(images, kernels, true_label):
    correct = 0
    total = len(images)

    # TODO: loop through the images
    # TODO: extract features
    # TODO: predict the class
    # TODO: compare prediction to true_label
    # TODO: count correct predictions

    accuracy = None
    return accuracy


# ============================================================
# SECTION 10: VISUALIZE RESULTS
#
# This helper function is already written for you.
# It shows:
#   - the original image
#   - the feature map
#   - the pooled map
# ============================================================

def show_results(image, feature_map, pooled_map, title=""):
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(feature_map, cmap="gray")
    axes[1].set_title("Feature Map")
    axes[1].axis("off")

    axes[2].imshow(pooled_map, cmap="gray")
    axes[2].set_title("Pooled Map")
    axes[2].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# SECTION 11: MAIN PROGRAM
#
# TASK:
# Complete the main function so that it:
#   1. loads circle images
#   2. loads square images
#   3. prints one example feature vector for a circle
#   4. prints one example feature vector for a square
#   5. evaluates accuracy on circles
#   6. evaluates accuracy on squares
#   7. prints overall accuracy
#   8. shows one example visualization
# ============================================================

def main():
    # TODO: load the two image sets
    circle_images, circle_names = [], []
    square_images, square_names = [], []

    # TODO: evaluate circles
    circle_accuracy = None

    # TODO: evaluate squares
    square_accuracy = None

    # TODO: compute overall accuracy
    overall_accuracy = None

    print("Circle accuracy:", circle_accuracy)
    print("Square accuracy:", square_accuracy)
    print("Overall accuracy:", overall_accuracy)

    # TODO: choose one image and one kernel
    # TODO: process that image
    # TODO: show original image, feature map, and pooled map


if __name__ == "__main__":
    main()