"""Contains function to return huberloss"""

import tensorflow as tf


def get_huber_loss_fn(**huber_loss_kwargs):
    """A function to return huberloss
    """
    def custom_huber_loss(y_true, y_pred):
        """Returns tensorflows huberloss with parameters already passed

        Args:
            y_true (_type_): y true values
            y_pred (_type_): y prediction values

        Returns:
            any: Huberloss function
        """
        return tf.compat.v1.losses.huber_loss(y_true, y_pred, **huber_loss_kwargs)
    return custom_huber_loss
