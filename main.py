import os
import tensorflow as tf
from tensorflow.keras import layers, models

base_dir = 'pictures'

# print out the name of each file in the directory
for root, dirs, files in os.walk(base_dir):
	for file in files:
		print(file)

# model = models.Sequential([
# 	# Convolutional base
# 	layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
# 	layers.MaxPooling2D((2, 2)),
# 	layers.Conv2D(64, (3, 3), activation='relu'),
# 	layers.MaxPooling2D((2, 2)),
# 	layers.Conv2D(128, (3, 3), activation='relu'),
# 	layers.MaxPooling2D((2, 2)),
# 	layers.Conv2D(128, (3, 3), activation='relu'),
# 	layers.MaxPooling2D((2, 2)),
#
# 	# Dense layers
# 	layers.Flatten(),
# 	layers.Dense(512, activation='relu'),
# 	layers.Dense(len(classes), activation='softmax')
# ])
