import inspect
#import rospy

def raiseNotDefined():
    fileName = inspect.stack()[1][1]
    line = inspect.stack()[1][2]
    method = inspect.stack()[1][3]

    rospy.loginfo("*** Method not implemented: %s at line %s of %s" % (method, line, fileName))



