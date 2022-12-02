"""Learning rate reducer script"""

import tensorflow as tf

class LearningRateReducerCb(tf.keras.callbacks.Callback):
    """A class to handle learning rate adjustments

    Args:
        tf (_type_): Inherits tensorflow callback
    """
    def on_epoch_end(self, epoch, logs={}):
        """Adjusts learning rate dynamically

        Args:
            epoch (_type_): self, inherit's class attributes.
            logs (dict, optional): Log data from training. Defaults to {}.
        """
        old_lr = self.model.optimizer.lr.read_value()
        new_lr = old_lr * 0.995
        print("\nEpoch: {}. Reducing Learning Rate from {} to {}".format(
            epoch, old_lr, new_lr))
        self.model.optimizer.lr.assign(new_lr)