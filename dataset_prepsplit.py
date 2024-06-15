from sklearn.model_selection import train_test_split
import os
import shutil

def trainTestSplit(augmentedFolderPath, saveDir, testSize : float = 0.25):
    '''
    Разбивает аугментированные данные из папки [augmentedFolderPath] в train и val в указанную папку \n
    augmentedFolderPath - папка откуда брать \n
    saveDir - папка куда сохранять \n
    testSize - соотношение тестовой выборки \n
    '''

    print("TTS started!")

    files : list[str] = os.listdir(augmentedFolderPath)
    imgs = list(filter(lambda it : it.endswith("jpg"), files))
    labels = list(filter(lambda it : it.endswith("txt"), files))

    print("augmented_imgs_len:", len(imgs))
    print("augmented_labels_len", len(labels))

    trainDir = os.path.join(saveDir, "train")
    valDir = os.path.join(saveDir, "val")
    trainImgsDir = os.path.join(trainDir, "images")
    trainLabelsDir = os.path.join(trainDir, "labels")
    valImgsDir = os.path.join(valDir, "images")
    valLabelsDir = os.path.join(valDir, "labels")

    try:
        os.mkdir(saveDir)
        os.mkdir(trainDir)
        os.mkdir(valDir)
        os.mkdir(trainImgsDir)
        os.mkdir(trainLabelsDir)
        os.mkdir(valImgsDir)
        os.mkdir(valLabelsDir)
    except: pass

    train_img, val_img, train_labels, val_labels = train_test_split(imgs, labels, test_size=testSize)
    
    for img in train_img: moveTo(img, augmentedFolderPath, trainImgsDir)
    for img in val_img: moveTo(img, augmentedFolderPath, valImgsDir)
    for label in train_labels: moveTo(label, augmentedFolderPath, trainLabelsDir)
    for label in val_labels: moveTo(label, augmentedFolderPath, valLabelsDir)


def moveTo(filename, fromFolder, toFolder):
    fromPath = os.path.join(fromFolder, filename)
    toPath = os.path.join(toFolder, filename)
    shutil.move(fromPath, toPath)
