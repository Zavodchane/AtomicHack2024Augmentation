import cv2
import matplotlib.pyplot as plt

from utils import readImage, readBBOXes, showImg, formatBBOXes

class Color:
    def red(): return (255, 0, 0)
    def white(): return (255, 255, 255)


def visualizeBBOX(img, bbox : tuple[float], className : str, color : tuple[int] = Color.red(), thickness : int = 2, fontScale = 1):
    '''
    Визуализация ограничивающих прямоугольников
    '''
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _ ) = cv2.getTextSize(className, cv2.FONT_HERSHEY_SIMPLEX, fontScale, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)

    cv2.putText(
        img, 
        text=className, 
        org=(x_min, y_min - int(0.3 * text_height)), 
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=fontScale,
        color=Color.white(),
        lineType=cv2.LINE_AA
    )
    
    return img


def visualize(image, bboxes, classNames):
    img = image.copy()
    for bbox, className in zip(bboxes, classNames):
        img = visualizeBBOX(img, bbox, className)
    
    return img


def actualizeCoordinates(imgWidth, imgHeight, bboxes):
    return list(
        map(
            lambda bbox: [
                (bbox[0] * imgWidth) - ((bbox[2] * imgWidth) / 2), 
                (bbox[1] * imgHeight) - ((bbox[3] * imgHeight) / 2), 
                bbox[2] * imgWidth, 
                bbox[3] * imgHeight
            ], bboxes
        )
    )


def visualizeYOLOLabel(img, bboxes):
    classNames = list(map(lambda arg: arg[-1], bboxes))
    height, width, _ = img.shape
    bboxes = actualizeCoordinates(width, height, bboxes)

    return visualize(img, bboxes, classNames)

    