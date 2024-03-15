# load_data.py
# load dicom images and save them as numpy

import os
import numpy as np
from medpy.io import load, save
import matplotlib.pyplot as plt




def load_images(folder_path):
    patient_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    patient_folders.sort()

    images = []
    for patient_folder in patient_folders:
        patient_image_folder = os.path.join(folder_path, patient_folder)
        dicom_files = [f for f in os.listdir(patient_image_folder) if f.endswith('.dcm')]

        patient_images = []
        for dicom_file in dicom_files:
            dicom_path = os.path.join(patient_image_folder, dicom_file)
            image, _ = load(dicom_path)
            patient_images.append(image)

        patient_images = np.array(patient_images)
        patient_images = np.squeeze(patient_images, axis=-1)
        patient_images = np.transpose(patient_images, (1, 2, 0))
        images.append(patient_images)

    np.save(os.path.join(folder_path, "images.npy"), np.array(images))

    return np.array(images)

def load_masks(folder_path):
    nrrd_files = [f for f in os.listdir(folder_path) if f.endswith('.nrrd')]
    nrrd_files.sort()

    masks = []
    for nrrd_file in nrrd_files:
        mask_path = os.path.join(folder_path, nrrd_file)
        mask, _ = load(mask_path)
        masks.append(np.array(mask))

    np.save(os.path.join(folder_path, "masks.npy"), np.array(masks))

    return np.array(masks)

# /datahome/mario/datasets/Images/NCI-ISBI-2013-Prostate-Challenge/Training_images/

train_images_folder = "/datahome/mario/datasets/Images/NCI-ISBI-2013-Prostate-Challenge/Training_images/"
train_masks_folder = "/datahome/mario/datasets/Images/NCI-ISBI-2013-Prostate-Challenge/Training_masks/"
train_images = load_images(train_images_folder)
train_masks = load_masks(train_masks_folder)

leaderboard_images_folder = "/content/drive/MyDrive/VCS Project/dataset/leaderboard/images"
leaderboard_masks_folder = "/content/drive/MyDrive/VCS Project/dataset/leaderboard/masks"
leaderboard_images = load_images(leaderboard_images_folder)
leaderboard_masks = load_masks(leaderboard_masks_folder)

test_images_folder = "/content/drive/MyDrive/VCS Project/dataset/test/images"
test_masks_folder = "/content/drive/MyDrive/VCS Project/dataset/test/masks"
test_images = load_images(test_images_folder)
test_masks = load_masks(test_masks_folder)