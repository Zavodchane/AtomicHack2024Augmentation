from augmentation import *
from dataset_prepsplit import trainTestSplit


augmentDatset("test_data", "augmentation_output")
trainTestSplit("augmentation_output", "augmented_dataset")