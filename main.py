import visualize
import utils
import matplotlib.pyplot as plt

img = utils.readImage("test/1 (1)_180_flip.jpg")
bboxes = utils.formatBBOXes(utils.readBBOXes("test/1 (1)_180_flip.txt"))

resImg = visualize.visualizeYOLOLabel(img, bboxes)
utils.showImg(resImg)
plt.show()