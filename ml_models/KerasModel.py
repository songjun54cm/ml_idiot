__author__ = 'JunSong<songjun54cm@gmail.com>'
from keras.models import Model
from keras import optimizers
import keras.backend as K
import logging


def get_optimizer(config):
    if(config['optimizer'] == 'rmsprop'):
        opti = optimizers.rmsprop(lr=config['learning_rate'],
                                  clipvalue=config['grad_clip'],
                                  decay=config['decay_rate'])
        return opti
    elif(config['optimizer'] == 'adadelta'):
        opti = optimizers.adadelta(lr=config['learning_rate'],
                                   clipvalue=config['grad_clip'])
        return opti
    elif(config['optimizer'] == 'sgd'):
        opti = optimizers.sgd(lr=config['learning_rate'],
                              momentum=config['momentum'],
                              decay=config['learning_rate_decay'])
        return opti
    else:
        raise KeyError('optimizer name error')


class KerasModel(Model):
    def __init__(self, inputs, outputs, name=None):
        self.save_ext = 'h5'
        super(KerasModel, self).__init__(inputs, outputs, name=name)

    def loss_pred_on_batch(self, x, y, sample_weight=None):
        """Test the model on a single batch of samples.

        # Arguments
            x: Numpy array of test data,
                or list of Numpy arrays if the model has multiple inputs.
                If all inputs in the model are named,
                you can also pass a dictionary
                mapping input names to Numpy arrays.
            y: Numpy array of target data,
                or list of Numpy arrays if the model has multiple outputs.
                If all outputs in the model are named,
                you can also pass a dictionary
                mapping output names to Numpy arrays.
            sample_weight: optional array of the same length as x, containing
                weights to apply to the model's loss for each sample.
                In the case of temporal data, you can pass a 2D array
                with shape (samples, sequence_length),
                to apply a different weight to every timestep of every sample.
                In this case you should make sure to specify
                sample_weight_mode="temporal" in compile().

        # Returns
            Scalar test loss (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs
            and/or metrics). The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.
        """
        x, y, sample_weights = self._standardize_user_data(
            x, y,
            sample_weight=sample_weight,
            check_batch_axis=True)
        if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
            ins = x + y + sample_weights + [0.]
        else:
            ins = x + y + sample_weights
        self._make_loss_pred_function()
        outputs = self.loss_pred_function(ins)
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def _make_loss_pred_function(self):
        if not hasattr(self, 'loss_pred_function'):
            raise RuntimeError('You must compile your model before using it.')
        if self.loss_pred_function is None:
            inputs = self._feed_inputs + self._feed_targets + self._feed_sample_weights
            if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
                inputs += [K.learning_phase()]
            # Return loss and metrics, no gradient updates.
            # Does update the network states.
            self.loss_pred_function = K.function(inputs,
                                            [self.total_loss] + self.outputs,
                                            updates=self.state_updates,
                                            **self._function_kwargs)

    def build_model(self, config):
        logging.info(self.summary())
        self.optimizer = get_optimizer(config)
        self.compile(optimizer=self.optimizer,
                     loss=config['loss'],
                     metrics=config['metrics'])
        self.loss_pred_function = None
        self._make_train_function()
        self._make_test_function()
        self._make_predict_function()
        self._make_loss_pred_function()

    def train_one_batch(self, batch_data):
        x = batch_data['x']
        y = batch_data['y']
        outs = self.train_on_batch(x, y)
        loss = outs[0]
        res = {
            'loss': loss
        }
        return res