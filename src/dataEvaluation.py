import numpy as np
import dataProcessor
import _mlpModel as model

import cv2
from skimage.transform  import resize

def evaluteData():
    imageArray, labelData = dataProcessor.importImages("/home/ambrosemcduffy/codingProjects/mlFromScratch/images/trainDataset")
    imageArrayTest, labelDataTest = dataProcessor.importImages("/home/ambrosemcduffy/codingProjects/mlFromScratch/images/testDataset")
    dataProcessor.displayImages(imageArrayTest, labelDataTest, size=(4,4))
    dataProcessor.displayImages(imageArray, labelData, size=(8, 8))


def predict(image, parameters):
    

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    size = dataProcessor.getNewImageSize(image, dims=[64,64, 3])
    newImage = resize(image, (size[0], size[1]))
    newImage = newImage[: 64, : 64]
    newImage[newImage < 1e-5] = 0

    imageFlat = newImage.reshape(1, -1).T
    imageFlat = imageFlat

    AL, caches = model.forward(imageFlat, parameters)
    
    threshold = 0.5
    yhat_binary = np.where(AL[0][0] >= threshold, 1, 0)
    return newImage, AL


def getAccuracy(a, y):
     return 100 - np.mean(np.abs(a - y)) * 100
