import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

(training_images, training_label), (testing_images, testing_label) = datasets.cifar10.load_data()


