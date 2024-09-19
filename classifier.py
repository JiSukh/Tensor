from tensorflow.keras import models
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

#Classifications which are classified within cifar10 dataset.
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog','horse', 'ship','truck']

model = models.load_model('image_classifier.keras')

img = cv.imread('images/plane.png')
img = cv.resize(img, (32, 32))  
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)


plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img])/255)
index = np.argmax(prediction)

print(f'prediction is {class_names[index]}')