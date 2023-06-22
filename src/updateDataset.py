import dataProcessor

# 
images, labels = dataProcessor.importImages()
dataProcessor.saveData(images, labels)
