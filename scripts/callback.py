"""Modified tensorflow callback class"""

import tensorflow as tf

class HaltCallback(tf.keras.callbacks.Callback):
    """A class to automatically halt training when val_loss has reached a specific criteria.

    Args:
        tf (_type_): Inherits tensorflow callback
    """

    def on_epoch_end(self, logs={}):
        """Halts training when condition met.

        Args:
            epoch: self, inherit's class attributes.
            logs (dict, optional): Log data from training. Defaults to {}.
        """
        if (logs.get('val_loss') <= 0.001):
            print("\n\n\nReached 0.05 loss value so cancelling training!\n\n\n")
            self.model.stop_training = True