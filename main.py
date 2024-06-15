from augmentation import augmentDataset
from dataset_prepsplit import trainTestSplit


augmentDataset("test_data", "augmentation_output")
trainTestSplit("augmentation_output", "augmented_dataset")