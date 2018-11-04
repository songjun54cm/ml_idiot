__author__ = 'JunSong<songjun54cm@gmail.com>'
import tensorflow as tf


class TFCNNModel(object):
    def __init__(self):
        self.input_label = None
        self.input_feature = None
        self.probabilities = None
        self.predict_class = None
        self.predict_accuracy = None
        self.model_loss = None
        self.regularize_loss = None
        self.total_loss = None

        self.train_op = None
        self.session = None

    def create(self, model_config):
        LearningRate = model_config['learning_rate']
        DropoutRate = model_config['dropout_rate']

        # Input Layer
        label_input_layer = tf.placeholder(tf.int64, [None], name='input_label')
        self.input_label = label_input_layer
        fea_input_layer = tf.placeholder(tf.float32, [None, 28, 28, 1], name='input_feature')
        self.input_feature = fea_input_layer

        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
            inputs=fea_input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Dense Layer
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense, rate=DropoutRate)

        logits = tf.layers.dense(inputs=dropout, units=10)

        probabilities = tf.nn.softmax(logits, name='softmax_tensor')
        self.probabilities = probabilities
        pred_class = tf.argmax(input=logits, axis=1, name='pred_class')
        self.predict_class = pred_class
        pred_accuracy = tf.reduce_mean(tf.to_float(tf.equal(pred_class, label_input_layer)), name='pred_accuracy')
        self.predict_accuracy = pred_accuracy

        # Calculate Loss
        model_loss = tf.losses.sparse_softmax_cross_entropy(labels=label_input_layer, logits=logits, name="model_loss")
        self.model_loss = model_loss

        total_loss = tf.add(model_loss, regularize_loss, name="total_loss")
        self.total_loss = total_loss

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss)
        self.train_op = train_op

        # # Set up logging for predictions
        # tensors_to_log = {"probabilities": "softmax_tensor"}
        # logging_hook = tf.train.LoggingTensorHook(
        #     tensors=tensors_to_log, every_n_iter=50)

        sess = tf.Session()
        self.session = sess
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

    def save(self, target_path):
        pass

    def load(self, soruce_path):
        pass