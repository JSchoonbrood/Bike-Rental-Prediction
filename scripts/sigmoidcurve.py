"""Contains function to return modified sigmoid curve"""

from keras import backend as K

def modified_sigmoid(x):
    """Returns a modified sigmoid curve

    Args:
        x (any): Tensor or variable taken from model.

    Returns:
        any: A tensor
    """
    return (K.sigmoid(x)-0.1)