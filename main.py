from augmentation import augmentDataset
from dataset_prepsplit import trainTestSplit


augmentDataset("dataset", "augmentation_output")
trainTestSplit("augmentation_output", "augmented_dataset")