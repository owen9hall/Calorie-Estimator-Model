from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers, models

# load InceptionV3 & exclude top layer (ANN)
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model

# add ANN layers
model = models.Sequential([
   base_model,
   layers.GlobalAveragePooling2D(),
   layers.Dropout(0.5),
   layers.Dense(101, activation='softmax')
])

model.summary()
model.save('food101_inceptionv3.keras')
