import albumentations as A

import utils
import os

def transformAndSave(transform, img, bboxes, imgFilename, bboxesFilename):
    transformResult = transform(image=img, bboxes=bboxes)
    augImg = transformResult["image"]
    augBboxes = transformResult["bboxes"]
    unformattedAugBboxes = utils.unformatBBOXes(augBboxes)
    unformattedAugBboxes = list(map(lambda uB: list(map(lambda b: str(b), uB)), unformattedAugBboxes))
    utils.saveBboxes(bboxesFilename, unformattedAugBboxes)
    utils.saveImage(imgFilename, augImg)


def augRot0(imgPath : str, labelsPath : str, resultFolderPath : str):
    img = utils.readImage(imgPath)
    bboxes = utils.readBBOXes(labelsPath)
    imgFilename = os.path.join(resultFolderPath, imgPath.split(os.path.sep)[-1].split(".")[0] + "_0.jpg")
    bboxesFilename = os.path.join(resultFolderPath, labelsPath.split(os.path.sep)[-1].split(".")[0] + "_0.txt")
    utils.saveBboxes(bboxesFilename, bboxes)
    utils.saveImage(imgFilename, img)


def augRot0Flip(imgPath : str, labelsPath : str, resultFolderPath : str):
    img = utils.readImage(imgPath)
    bboxes = utils.formatBBOXes(utils.readBBOXes(labelsPath))
    transform = A.Compose([
        A.HorizontalFlip(
            always_apply=True,
            p = 1
        ),
        A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=0, p=0.7)
    ], bbox_params=A.BboxParams(format="yolo", min_visibility=0.6))
    imgFilename = os.path.join(resultFolderPath, imgPath.split(os.path.sep)[-1].split(".")[0] + "_0_flip.jpg")
    bboxesFilename = os.path.join(resultFolderPath, labelsPath.split(os.path.sep)[-1].split(".")[0] + "_0_flip.txt")
    transformAndSave(transform, img, bboxes, imgFilename, bboxesFilename)


def augRot90(imgPath : str, labelsPath : str, resultFolderPath : str):
    img = utils.readImage(imgPath)
    height, width, _ = img.shape
    bboxes = utils.formatBBOXes(utils.readBBOXes(labelsPath))
    transform = A.Compose([
        A.Resize(height=width, width=height, always_apply=True),
        A.SafeRotate(
            [-90, -90],
            p = 1,
        ),
        A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=0, p=0.7)
    ], bbox_params=A.BboxParams(format="yolo", min_visibility=0.6))
    imgFilename = os.path.join(resultFolderPath, imgPath.split(os.path.sep)[-1].split(".")[0] + "_90.jpg")
    bboxesFilename = os.path.join(resultFolderPath, labelsPath.split(os.path.sep)[-1].split(".")[0] + "_90.txt")
    transformAndSave(transform, img, bboxes, imgFilename, bboxesFilename)


def augRot90Flip(imgPath : str, labelsPath : str, resultFolderPath : str):
    img = utils.readImage(imgPath)
    height, width, _ = img.shape
    bboxes = utils.formatBBOXes(utils.readBBOXes(labelsPath))
    transform = A.Compose([
        A.Resize(height=width, width=height, always_apply=True),
        A.SafeRotate(
            [-90, -90],
            p = 1,
        ),
        A.HorizontalFlip(
            always_apply=True,
            p = 1
        ),
        A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=0, p=0.7)
    ], bbox_params=A.BboxParams(format="yolo", min_visibility=0.6))
    imgFilename = os.path.join(resultFolderPath, imgPath.split(os.path.sep)[-1].split(".")[0] + "_90_flip.jpg")
    bboxesFilename = os.path.join(resultFolderPath, labelsPath.split(os.path.sep)[-1].split(".")[0] + "_90_flip.txt")
    transformAndSave(transform, img, bboxes, imgFilename, bboxesFilename)


def augRot180(imgPath : str, labelsPath : str, resultFolderPath : str):
    img = utils.readImage(imgPath)
    height, width, _ = img.shape
    bboxes = utils.formatBBOXes(utils.readBBOXes(labelsPath))
    transform = A.Compose([
        A.Resize(height=height, width=width, always_apply=True),
        A.SafeRotate(
            [-180, -180],
            p = 1,
        ),
        A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=0, p=0.7)
    ], bbox_params=A.BboxParams(format="yolo", min_visibility=0.6))
    imgFilename = os.path.join(resultFolderPath, imgPath.split(os.path.sep)[-1].split(".")[0] + "_180.jpg")
    bboxesFilename = os.path.join(resultFolderPath, labelsPath.split(os.path.sep)[-1].split(".")[0] + "_180.txt")
    transformAndSave(transform, img, bboxes, imgFilename, bboxesFilename)


def augRot180Flip(imgPath : str, labelsPath : str, resultFolderPath : str):
    img = utils.readImage(imgPath)
    height, width, _ = img.shape
    bboxes = utils.formatBBOXes(utils.readBBOXes(labelsPath))
    transform = A.Compose([
        A.Resize(height=height, width=width, always_apply=True),
        A.SafeRotate(
            [-180, -180],
            p = 1,
        ),
        A.HorizontalFlip(
            always_apply=True,
            p = 1
        ),
        A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=0, p=0.7)
    ], bbox_params=A.BboxParams(format="yolo", min_visibility=0.6))
    imgFilename = os.path.join(resultFolderPath, imgPath.split(os.path.sep)[-1].split(".")[0] + "_180_flip.jpg")
    bboxesFilename = os.path.join(resultFolderPath, labelsPath.split(os.path.sep)[-1].split(".")[0] + "_180_flip.txt")
    transformAndSave(transform, img, bboxes, imgFilename, bboxesFilename)


def augRot270(imgPath : str, labelsPath : str, resultFolderPath : str):
    img = utils.readImage(imgPath)
    height, width, _ = img.shape
    bboxes = utils.formatBBOXes(utils.readBBOXes(labelsPath))
    transform = A.Compose([
        A.Resize(height=width, width=height, always_apply=True),
        A.SafeRotate(
            [-270, -270],
            p = 1,
        ),
        A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=0, p=0.7)
    ], bbox_params=A.BboxParams(format="yolo", min_visibility=0.6))
    imgFilename = os.path.join(resultFolderPath, imgPath.split(os.path.sep)[-1].split(".")[0] + "_270.jpg")
    bboxesFilename = os.path.join(resultFolderPath, labelsPath.split(os.path.sep)[-1].split(".")[0] + "_270.txt")
    transformAndSave(transform, img, bboxes, imgFilename, bboxesFilename)


def augRot270Flip(imgPath : str, labelsPath : str, resultFolderPath : str):
    img = utils.readImage(imgPath)
    height, width, _ = img.shape
    bboxes = utils.formatBBOXes(utils.readBBOXes(labelsPath))
    transform = A.Compose([
        A.Resize(height=width, width=height, always_apply=True),
        A.SafeRotate(
            [-270, -270],
            p = 1,
        ),
        A.HorizontalFlip(
            always_apply=True,
            p = 1
        ),
        A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=0, p=0.7)
    ], bbox_params=A.BboxParams(format="yolo", min_visibility=0.6))
    imgFilename = os.path.join(resultFolderPath, imgPath.split(os.path.sep)[-1].split(".")[0] + "_270_flip.jpg")
    bboxesFilename = os.path.join(resultFolderPath, labelsPath.split(os.path.sep)[-1].split(".")[0] + "_270_flip.txt")
    transformAndSave(transform, img, bboxes, imgFilename, bboxesFilename)


def augmentDatset(datasetFolder, resultFolder):
    print("Augmentation started!")

    try: os.mkdir(resultFolder)
    except: pass

    imgs = list(filter(lambda it: it.endswith(".jpg"), os.listdir(datasetFolder)))
    labels = list(filter(lambda it: it.endswith(".txt"), os.listdir(datasetFolder)))

    imgs_paths = list(map(lambda arg: os.path.join(datasetFolder, arg), imgs))
    labels_paths = list(map(lambda arg: os.path.join(datasetFolder, arg), labels))

    for img, label in zip(imgs_paths, labels_paths):
        print(f"augmenting: {img} and {label}")
        augRot0(img, label, resultFolder)
        augRot0Flip(img, label, resultFolder)
        augRot90(img, label, resultFolder)
        augRot90Flip(img, label, resultFolder)
        augRot180(img, label, resultFolder)
        augRot180Flip(img, label, resultFolder)
        augRot270(img, label, resultFolder)
        augRot270Flip(img, label, resultFolder)