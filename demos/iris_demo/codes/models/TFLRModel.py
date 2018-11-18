__author__ = 'JunSong<songjun54cm@gmail.com>'
import argparse
import tensorflow as tf
from ml_idiot.ml_models.TensorFlowModel import TensorFlowModel


class TFLRModel(TensorFlowModel):
    def __init__(self, config):
        super(TensorFlowModel, self).__init__(config)
        self.session = None
        self.loss = None

    def create(self, config):
        self.x = tf.placeholder(tf.float32, [None, config["fea_size"]])
        self.y = tf.placeholder(tf.float32, [None, 1])
        self.w = tf.get_variable("w", (config["fea_size"], 1), dtype=tf.float32)
        self.b = tf.get_variable("b", (1,), dtype=tf.float32)

        self.s = tf.matmul(self.x, self.w) + self.b
        self.e = tf.exp(self.s)
        self.pred = tf.divide(self.e, tf.add(self.e, 1.0))
        self.pred_y = tf.greater_equal(self.pred, 0.5)
        self.loss = tf.reduce_mean(tf.log(1.0 + self.e) - (self.y * self.s))

        self.optimizer = self.get_optimizer(config)
        self.train_op = self.optimizer.minimize(self.loss)

        sess = tf.Session()
        self.session = sess
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)


    def train_batch(self, batch_data, optimizer=None):
        if self.session is None:
            sess = tf.Session()
            self.session = sess
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)
        batch_x = batch_data["x"]
        batch_y = batch_data["y"]
        _, batch_loss = self.session.run([self.train_op, self.loss],
                                   feed_dict={
                                    self.x: batch_x,
                                    self.y: batch_y
                                   })
        train_res = {
            "batch_loss": batch_loss,
            "score_loss": batch_loss,
            "regu_loss": batch_loss,
        }
        return train_res

    def predict_batch(self, batch_data):
        batch_x = batch_data["x"]
        batch_y = batch_data["y"]
        loss, pred_vals = self.session.run([self.loss,self.pred_y],
                                           feed_dict={
                                             self.x: batch_x,
                                             self.y: batch_y
                                            })
        res = {
            "loss": loss,
            "pred_vals": pred_vals,
            "gth_vals": batch_y
        }
        return res

    def save(self, file_path):
        pass