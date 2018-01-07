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

def set_params(params, namespace=""):
    """create an object from a list of params for use within an object
       Warning: only for use with parameters which do not change at runtime
       changes to parameters will not be made to the dict.
       Params
       --- 
       params: a list of strings representing the parameter to be aquired
       namespace: the namespace for which the pram exists e.g. /control_vars/P
       Returns
       ---
       pram_dict: a dict populated with the parameters from the parameter server
       Example usage
       ---
        params = ["dt","mass"]
        namespace = "/human_model/"
        self.params = util.set_params(params,namespace)
    """
    param_dict = {}
    for param in params:
        print(namespace+param)
        param_dict[param] = rospy.get_param(namespace+param)
    return param_dict