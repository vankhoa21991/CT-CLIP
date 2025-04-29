import os
import nibabel as nib
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import glob
from multiprocessing import Pool
from tqdm import tqdm

def read_nii_files(directory):
    """
    Retrieve paths of all NIfTI files in the given directory.

    Args:
    directory (str): Path to the directory containing NIfTI files.

    Returns:
    list: List of paths to NIfTI files.
    """

    nii_files = []
    for root, dirs, files in tqdm(os.walk(directory)):
        for file in files:
            if file.endswith('.npz'):
                nii_files.append(os.path.join(root, file))

    print(f"Found {len(nii_files)} NIfTI files in {directory}")
    print(nii_files[:5])  # Print first 5 files for verification
    return nii_files


# Example usage:
if __name__ == "__main__":
    split_to_preprocess = '/mnt/datalake/DS-lake/vankhoa/CT-RATE/dataset/train_preprocessed' #select the validation or test split
    nii_files = read_nii_files(split_to_preprocess)
