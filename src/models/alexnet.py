import tensorflow as tf

# build AlexNet model
def build_model(height, width, channels, actions):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, activation='relu', input_shape=(height, width, channels)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=3, strides=2))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=5, strides=1, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=3, strides=2))
    model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=3, strides=1, activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=3, strides=1, activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=3, strides=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dense(actions, activation='linear'))
    return model