import cv2
import matplotlib.pyplot as plt


classIdToClassName = {
    "0" : "adj",
    "1" : "int",
    "2" : "geo",
    "3" : "pro",
    "4" : "non"
}

classNameToClassId = {
    "adj" : "0",
    "int" : "1",
    "geo" : "2",
    "pro" : "3",
    "non" : "4"
}


def readImage(imgPath):
    image = cv2.imread(imgPath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def saveImage(saveImgPath, img):
    cv2.imwrite(saveImgPath, img)

def saveBboxes(saveBboxesPath, bboxes):
    strBboxes = list(map(lambda bbox: unformattedBboxToString(bbox), bboxes))
    with open(saveBboxesPath, "w") as bboxes:
        bboxes.writelines(strBboxes)

def readBBOXes(labelsPath) -> list:
    bboxes = []

    with open(labelsPath) as labels:
        labelLines = labels.readlines()
    
    for idx in range(len(labelLines)):
        labelLines[idx] = labelLines[idx].replace("\n", "")
        classId, x_min, y_min, w, h = labelLines[idx].split(" ")
        bboxes.append([classId, x_min, y_min, w, h])

    return bboxes


def showImg(img):
    plt.figure(figsize=(16,9))
    plt.axis('off')
    plt.imshow(img)


def formatBBOXes(bboxesUnformatted):
    formattedBBOXes = []

    for bbox in bboxesUnformatted:
        formattedBBOXnoLabel = list(map(lambda coord: float(coord), bbox[1::]))
        label = classIdToClassName[bbox[0]]
        formattedBBOX = [*formattedBBOXnoLabel, label]
        formattedBBOXes.append(formattedBBOX)

    return formattedBBOXes

def unformatBBOXes(bboxesFormatted):
    unformattedBBOXes = []

    for bbox in bboxesFormatted:
        label = classNameToClassId[bbox[-1]]
        unformattedBBOX = [label, *bbox[:-1]]
        unformattedBBOXes.append(unformattedBBOX)

    return unformattedBBOXes


def unformattedBboxToString(unformattedBbox : list) -> str:
    
    return " ".join(unformattedBbox) + "\n"
