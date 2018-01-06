import inspect
import rospy
import numpy as np

def raiseNotDefined():
    fileName = inspect.stack()[1][1]
    line = inspect.stack()[1][2]
    method = inspect.stack()[1][3]

    rospy.loginfo("*** Method not implemented: %s at line %s of %s" % (method, line, fileName))


def direction(x, y):
    """ Compute the direction from x to y
        Params
        ---
        x: a numpy array
        y: a numpy array
        Returns
        ---
        normalized direction: numpy array of unit length to y from x
    """
    return (y - x)/np.linalg.norm(y - x + 1e-12)

def normalize(beliefs):
    """ Normalizes a list of scaler beliefs
        Params
        ---
        beliefs: a list of scaler beliefs #TODO check this
        Returns
        ---
        norm_beliefs: a np array of normalized beliefs
    """
    return np.asarray(beliefs)/np.sum(np.asarray(beliefs))
