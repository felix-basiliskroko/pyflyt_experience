import numpy as np


def ang(v1, v2):
    """
    Calculate the angle between two normalised vectors. Used to compute the angle between the line of sight and the target.
    :param v1:  Vector 1
    :param v2:  Vector 2
    :return:  Angle between the two vectors
    """
    angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return angle
