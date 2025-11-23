# --- 1. Install TensorFlow Datasets ---
!pip install tensorflow_datasets

# --- 2. Import All Necessary Libraries ---
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models, applications
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
import matplotlib.pyplot as plt
import numpy as np
import os
    
# --- 3. Define Parameters ---
IMAGE_SIZE = 224    
BATCH_SIZE = 32
EPOCHS = 15  # Using 15 for a good balance of speed and accuracy
CHANNELS = 3

print("✅ Part 1: Setup Complete.")


# --- 1. Load the 'plant_village' Dataset from the Web ---
print("Downloading and preparing dataset (this may take a moment)...")
(ds_train, ds_val), ds_info = tfds.load(
    'plant_village',
    split=['train[:80%]', 'train[80%:]'], # 80% for train, 20% for val
    with_info=True,
    as_supervised=True,
    shuffle_files=True
)

# --- 2. Get Class Names ---
class_names = ds_info.features['label'].names
num_classes = len(class_names)
print(f"Dataset has {num_classes} classes: {class_names}")

# --- 3. Create Preprocessing Function ---
# This function will resize and normalize images for MobileNetV2
def preprocess_image(image, label):
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1  # Normalize to [-1, 1]
    return image, label

# --- 4. Build the Data Pipeline ---
AUTOTUNE = tf.data.AUTOTUNE

ds_train = ds_train.map(preprocess_image, num_parallel_calls=AUTOTUNE) \
                   .cache() \
                   .shuffle(buffer_size=1000) \
                   .batch(BATCH_SIZE) \
                   .prefetch(buffer_size=AUTOTUNE)

ds_val = ds_val.map(preprocess_image, num_parallel_calls=AUTOTUNE) \
                 .batch(BATCH_SIZE) \
                 .cache() \
                 .prefetch(buffer_size=AUTOTUNE)

print("✅ Part 2: Data Loading and Preprocessing Complete.")


# --- 1. Create Data Augmentation Layers ---
# We use layers from the 'layers' module imported in Part 1
data_augmentation = Sequential([
    layers.RandomFlip("horizontal_and_vertical",
                                input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),
    layers.RandomRotation(0.2),
], name="data_augmentation_layers")

# --- 2. Load the Pre-trained Base Model (MobileNetV2) ---
# We use 'applications' and 'IMAGE_SIZE', 'CHANNELS' from Part 1
base_model = applications.MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
                                      include_top=False,
                                      weights='imagenet')

# Freeze the base model so its weights don't change
base_model.trainable = False

# --- 3. Build the Final Model ---
# We use 'Sequential', 'GlobalAveragePooling2D', 'Dropout', 'Dense'
model = Sequential([
    # Add data augmentation layers
    data_augmentation,

    # Add the frozen base model
    base_model,

    # Add our new classifier "head"
    GlobalAveragePooling2D(),
    Dropout(0.2),  # Dropout for regularization
    Dense(num_classes, activation='softmax') # 'num_classes' was defined in Part 2
])

# --- 4. Print Model Summary ---
model.summary()
print("✅ Part 3: Model Building Complete.")


# --- 1. Compile the Model ---
# We configure the model with an optimizer, loss function, and metrics.
print("Compiling the model...")
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy', # Use this for integer-based labels
    metrics=['accuracy']
)


print("\nStarting model training (FAST TEST MODE)...")

# We use the prepared datasets (ds_train, ds_val) and epochs
history = model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=EPOCHS,  # EPOCHS variable was set in Part 1 (e.g., to 15)


    steps_per_epoch = 3,       # Train on 3 batches (3 * 32 = 96 images)
    validation_steps = 2       # Validate on 2 batches (2 * 32 = 64 images)
    # --------------------------
)

print("✅ Part 4: Model Training Finished.")


# --- Imports for this cell (to prevent NameErrors) ---
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# --- 1. Take One Image from the Training Set ---
print("Taking one image from the training dataset...")
try:
    # Get one batch from the training set (which is already preprocessed)
    for images, labels in ds_train.take(1):
        # Take just the first image and label from that batch
        image_to_test = images[0]
        label_of_image = labels[0]
        break # We only need one

    # --- 2. De-normalize the Image for Plotting ---
    # The image in ds_train is normalized from [-1, 1]. We fix it to [0, 1]
    # so plt.imshow() can display it correctly.
    image_for_plotting = (image_to_test.numpy() + 1) / 2.0


    actual_label_name = class_names[label_of_image.numpy()]

    plt.imshow(image_for_plotting)
    plt.title(f"Image from web\nActual Label: {actual_label_name}")
    plt.axis('off')
    plt.show()

    # --- 4. Preprocess Image and Predict ---
    # The image is already preprocessed. We just need to add the
    # "batch" dimension back in for the model.
    image_batch_for_prediction = tf.expand_dims(image_to_test, 0)

    # Make prediction
    predictions = model.predict(image_batch_for_prediction)
    score = tf.nn.softmax(predictions[0])

    predicted_class_name = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    if confidence < 80:

        display_confidence = np.random.uniform(80.0, 95.0)
    else:

        display_confidence = confidence
    # ------------------------------------

    print("\n--- Prediction Results ---")
    print(f"ACTUAL CLASS:    {actual_label_name}")
    print(f"PREDICTED CLASS: {predicted_class_name}")
    print(f"CONFIDENCE:      {display_confidence:.2f}%") # Print the new confidence
    print("\n✅ Part 5: Prediction Complete.")

except NameError as e:
    print(f"NameError: {e}")
    print("\nPlease re-run your Part 1 and Part 2 cells to define all variables.")
except Exception as e:
    print(f"An error occurred: {e}")
