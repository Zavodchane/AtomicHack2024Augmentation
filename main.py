from augmentation import augmentDataset
from dataset_prepsplit import trainTestSplit
from visualize import visualizeYOLOLabel
import utils
import matplotlib.pyplot as plt 


augmentDataset("dataset", "augmentation_output")
trainTestSplit("augmentation_output", "augmented_dataset")

# img = utils.readImage("debug/3 (34).jpg")
# labels = utils.formatBBOXes(utils.readBBOXes("debug/3 (34).txt"))

# resImg = visualizeYOLOLabel(img, labels)
# utils.showImg(resImg)
# plt.show()