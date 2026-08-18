import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

NUM_CLASSES = 43
IMAGE_SIZE = (32, 32)

# ---------------- CNN Model ---------------- #

def build_cnn():
    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),

        layers.Conv2D(32, (3,3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3,3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3,3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3,3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),

        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    return model


# ---------------- Transfer Learning ---------------- #

def build_transfer_model():

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(32,32,3)
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(32,32,3))

    x = base_model(inputs, training=False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    return model


# ---------------- Train Function ---------------- #

def train_model(model, x_train, y_train, x_test, y_test):

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    )

    datagen = ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        horizontal_flip=False,
        fill_mode="nearest"
    )

    datagen.fit(x_train)

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=64),
        validation_data=(x_test, y_test),
        epochs=30,
        callbacks=[early_stop],
        verbose=1
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    print(f"Test Accuracy : {accuracy:.4f}")

    return model, history

#---------------------mapping---------------
sign_classes = {
    0: 'Speed limit (20km/h)',
        1: 'Speed limit (30km/h)',
        2: 'Speed limit (50km/h)',  
        3: 'Speed limit (60km/h)',
        4: 'Speed limit (70km/h)',  
        5: 'Speed limit (80km/h)',
        6: 'End of speed limit (80km/h)',
        7: 'Speed limit (100km/h)',
        8: 'Speed limit (120km/h)',
        9: 'No passing',
        10: 'No passing for vehicles over 3.5 metric tons',
        11: 'Right-of-way at the next intersection',
        12: 'Priority road',
        13: 'Yield',
        14: 'Stop',
        15: 'No vehicles',
        16: 'Vehicles over 3.5 metric tons prohibited',
        17: 'No entry',
        18: 'General caution',
        19: 'Dangerous curve to the left',
        20: 'Dangerous curve to the right',
        21: 'Double curve', 
        22: 'Bumpy road',
        23: 'Slippery road',
        24: 'Road narrows on the right',
        25: 'Road work',
        26: 'Traffic signals',
        27: 'Pedestrians',
        28: 'Children crossing',
        29: 'Bicycles crossing',
        30: 'Beware of ice/snow',
        31: 'Wild animals crossing',
        32: 'End of all speed and passing limits',
        33: 'Turn right ahead',
        34: 'Turn left ahead',
        35: 'Ahead only',
        36: 'Go straight or right',
        37: 'Go straight or left',
        38: 'Keep right',
        39: 'Keep left',
        40: 'Roundabout mandatory',
        41: 'End of no passing',
        42: 'End of no passing by vehicles over 3.5 metric tons'
    
 }
