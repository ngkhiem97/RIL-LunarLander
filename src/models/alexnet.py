import tensorflow as tf

# build AlexNet model
def build_model(input_data, num_classes, keep_prob):
    # first convolutional layer
    conv1 = tf.layers.conv2d(inputs=input_data, filters=96, kernel_size=[11, 11], strides=4, padding='valid', activation=tf.nn.relu)
    # max pooling layer
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)
    # second convolutional layer
    conv2 = tf.layers.conv2d(inputs=pool1, filters=256, kernel_size=[5, 5], padding='same', activation=tf.nn.relu)
    # max pooling layer
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)
    # third convolutional layer
    conv3 = tf.layers.conv2d(inputs=pool2, filters=384, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
    # fourth convolutional layer
    conv4 = tf.layers.conv2d(inputs=conv3, filters=384, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
    # fifth convolutional layer
    conv5 = tf.layers.conv2d(inputs=conv4, filters=256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
    # max pooling layer
    pool3 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3, 3], strides=2)
    # flatten layer
    flat = tf.reshape(pool3, [-1, 6 * 6 * 256])
    # dropout layer
    dropout = tf.layers.dropout(inputs=flat, rate=keep_prob)
    # fully connected layer
    fc1 = tf.layers.dense(inputs=dropout, units=4096, activation=tf.nn.relu)
    # dropout layer
    dropout2 = tf.layers.dropout(inputs=fc1, rate=keep_prob)
    # fully connected layer
    fc2 = tf.layers.dense(inputs=dropout2, units=4096, activation=tf.nn.relu)
    # last fully connected layer
    output = tf.layers.dense(inputs=fc2, units=num_classes)
    return output