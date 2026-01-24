#!/usr/bin/env python
"""
Generate train/dev/test split files for Semantic2D dataset.
Automatically calculates the number of samples based on percentages.
"""
import os
from os import listdir, getcwd
from os.path import join, expanduser
import random

if __name__ == '__main__':
    ################ CUSTOMIZATION REQUIRED ################
    # The path of your dataset folder:
    train_folder = '~/semantic2d_data/2024-04-11-15-24-29'

    # Split percentages (must sum to 1.0)
    TRAIN_RATIO = 0.70  # 70% for training
    DEV_RATIO = 0.10    # 10% for validation/development
    TEST_RATIO = 0.20   # 20% for testing
    ########################################################

    # Expand user path (handle ~)
    train_folder = expanduser(train_folder)

    # Verify ratios sum to 1.0
    assert abs(TRAIN_RATIO + DEV_RATIO + TEST_RATIO - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {TRAIN_RATIO + DEV_RATIO + TEST_RATIO}"

    # The index files of datasets:
    train_txt = train_folder + '/train.txt'
    dev_txt = train_folder + '/dev.txt'
    test_txt = train_folder + '/test.txt'

    # Get the list of data files:
    positions_folder = train_folder + '/positions'
    if not os.path.exists(positions_folder):
        print(f"Error: Folder not found: {positions_folder}")
        print("Please check the train_folder path.")
        exit(1)

    train_files = os.listdir(positions_folder)

    # Filter only .npy files
    train_list = [f for f in train_files if f.endswith(".npy")]

    # Sort the list according to the name without extension:
    train_list.sort(key=lambda x: int(x[:-4]))

    # Shuffle the list
    random.shuffle(train_list)

    # Calculate split sizes based on percentages
    total_samples = len(train_list)
    NUM_TRAIN = int(total_samples * TRAIN_RATIO)
    NUM_DEV = int(total_samples * DEV_RATIO)
    NUM_TEST = total_samples - NUM_TRAIN - NUM_DEV  # Remaining samples go to test

    print(f"Dataset folder: {train_folder}")
    print(f"Total samples: {total_samples}")
    print(f"Split ratios: Train={TRAIN_RATIO:.0%}, Dev={DEV_RATIO:.0%}, Test={TEST_RATIO:.0%}")
    print(f"Split sizes:  Train={NUM_TRAIN}, Dev={NUM_DEV}, Test={NUM_TEST}")

    # Open txt files:
    train_file = open(train_txt, 'w')
    dev_file = open(dev_txt, 'w')
    test_file = open(test_txt, 'w')

    # Write to txt files based on calculated splits:
    for idx, file_name in enumerate(train_list):
        if idx < NUM_TRAIN:  # train
            train_file.write(file_name + '\n')
        elif idx < NUM_TRAIN + NUM_DEV:  # dev
            dev_file.write(file_name + '\n')
        else:  # test
            test_file.write(file_name + '\n')

    train_file.close()
    dev_file.close()
    test_file.close()

    print(f"\nGenerated split files:")
    print(f"  - {train_txt}")
    print(f"  - {dev_txt}")
    print(f"  - {test_txt}")
    print("Done!")
