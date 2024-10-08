
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

#load and prepare dataset
(training_images, training_label), (testing_images, testing_label) = datasets.cifar10.load_data()
training_images = training_images / 255
testing_images = testing_images / 255



for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_label[i][0]])

plt.show()

model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape = (32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())

model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10, activation = 'softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics =['accuracy'])
model.fit(training_images, training_label, epochs = 10, validation_data=(testing_images, testing_label))

loss, accuracy = model.evaluate(testing_images, testing_label)

print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

model.save('image_classifier.keras')

