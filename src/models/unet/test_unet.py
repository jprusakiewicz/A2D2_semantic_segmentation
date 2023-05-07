import tensorflow as tf


# Define the U-Net model architecture
def unet_model(input_shape, num_classes):
    """model: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/"""

    inputs = tf.keras.layers.Input(shape=input_shape)

    inputs_padded = tf.keras.layers.ZeroPadding2D(((0, 105), (0, 16)))(inputs)

    # region Encoder
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        inputs_padded)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = tf.keras.layers.BatchNormalization()(conv4)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = tf.keras.layers.BatchNormalization()(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    # lowest layer
    conv5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = tf.keras.layers.BatchNormalization()(conv5)
    conv5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = tf.keras.layers.BatchNormalization()(conv5)
    drop5 = tf.keras.layers.Dropout(0.5)(conv5)
    # endregion

    # region Decoder
    up1 = tf.keras.layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(drop5)
    merge1 = tf.keras.layers.concatenate([conv4, up1], axis=3)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge1)
    conv6 = tf.keras.layers.BatchNormalization()(conv6)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = tf.keras.layers.BatchNormalization()(conv6)

    up2 = tf.keras.layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv6)
    merge2 = tf.keras.layers.concatenate([conv3, up2], axis=3)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge2)
    conv7 = tf.keras.layers.BatchNormalization()(conv7)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = tf.keras.layers.BatchNormalization()(conv7)

    up3 = tf.keras.layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv7)
    merge3 = tf.keras.layers.concatenate([conv2, up3], axis=3)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge3)
    conv8 = tf.keras.layers.BatchNormalization()(conv8)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = tf.keras.layers.BatchNormalization()(conv8)

    up4 = tf.keras.layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv8)
    merge4 = tf.keras.layers.concatenate([conv1, up4], axis=3)

    conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge4)
    conv9 = tf.keras.layers.BatchNormalization()(conv9)
    conv9 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = tf.keras.layers.BatchNormalization()(conv9)
    conv9 = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = tf.keras.layers.BatchNormalization()(conv9)
    # endregion

    # Output
    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation='sigmoid')(conv9)

    outputs_unpadded = tf.keras.layers.Cropping2D(cropping=((0, 105), (0, 16)))(outputs)  # todo check if legit

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs_unpadded)

    return model
