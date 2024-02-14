import numpy as np
import os
from PIL import Image


dataset_path = "C:\\Users\\Nick\\Repos\\EEG2Image\\eeg2_data"
subjects = ["sub-01"]
# subjects = [
#         # "sub-01",
#         # "sub-02",
#         # "sub-03",
#         # "sub-04",
#         # "sub-05",
#         # "sub-06",
#         # "sub-07",
#         # "sub-08",
#         # "sub-09", 
#         # "sub-10",
#     ]  # 20GB

def create_EEG_dataset(path, subjects):
    subject_eeg_train_data = {}
    subject_eeg_test_data = {}
    # Obtains data per subject and adds it into the dictionary
    for subject in subjects:
        print("Loading", subject)
        eeg_parent_dir = os.path.join(path, 'preprocessed_data', subject)
        eeg_data_train = np.load(os.path.join(eeg_parent_dir, 'preprocessed_eeg_training.npy'), allow_pickle=True).item()
        print(eeg_data_train.keys())
        eeg_data_test = np.load(os.path.join(eeg_parent_dir, 'preprocessed_eeg_test.npy'), allow_pickle=True).item()
        subject_eeg_train_data[subject] = eeg_data_train
        subject_eeg_test_data[subject] = eeg_data_test

        print('Training EEG data shape per subject:')
        print(eeg_data_train['preprocessed_eeg_data'].shape, '- (Training image conditions × Training EEG repetitions × EEG channels × '
            'EEG time points)')
        print('Test EEG data shape per subject:')
        print(eeg_data_test['preprocessed_eeg_data'].shape, '- (Test image conditions × Test EEG repetitions × EEG channels × '
            'EEG time points)\n')
    
        
    return subject_eeg_train_data, subject_eeg_test_data

def create_image_dataset(path):
    img_parent_dir  = os.path.join(dataset_path, 'image_set')
    img_metadata = np.load(os.path.join(img_parent_dir, 'image_metadata.npy'),
        allow_pickle=True).item()
    
    train_images = []
    for train_img_idx in range(len(img_metadata['train_img_files'])):
        dataset_img_dir = os.path.join(img_parent_dir, 'training_images', img_metadata['train_img_concepts'][train_img_idx], img_metadata['train_img_files'][train_img_idx])
        dataset_img = Image.open(dataset_img_dir).convert('RGB')
        train_images.append(dataset_img)
    
    test_images = []
    for test_img_idx in range(len(img_metadata['test_img_files'])):
        dataset_img_dir = os.path.join(img_parent_dir, 'test_images', img_metadata['test_img_concepts'][test_img_idx], img_metadata['test_img_files'][test_img_idx])
        dataset_img = Image.open(dataset_img_dir).convert('RGB')
        test_images.append(dataset_img)

    # 10 images per concept.
    n_train_img = len(img_metadata['train_img_concepts'])
    n_train_concepts = len(np.unique(img_metadata['train_img_concepts']))
    n_train_img_per_concept = int(n_train_img / n_train_concepts)
    print('Training images: ' + str(n_train_img))
    print('Training image concepts: ' + str(n_train_concepts))
    print('Training images per concept: '+ str(n_train_img_per_concept))

    # 1 image per concept.
    n_test_img = len(img_metadata['test_img_concepts'])
    n_test_concepts = len(np.unique(img_metadata['test_img_concepts']))
    n_test_img_per_concept = int(n_test_img / n_test_concepts)
    print('Test images: ' + str(n_test_img))
    print('Test image concepts: ' + str(n_test_concepts))
    print('Test images per concept: ' + str(n_test_img_per_concept))
    return train_images, test_images

