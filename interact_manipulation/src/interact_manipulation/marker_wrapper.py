from visualization_msgs.msg import *
import rospy
import random
from geometry_msgs.msg import Point, Pose, Quaternion

marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
marker_list = []

# class MarkerWrapper():
#     def __init__(self, position, **kwargs):
#         self.kwarg = kwargs
#         self.position = position
    
#     @property
#     def marker(self):
#         """Return a marker object"""
#         #if position in positions
#         num_markers = len(marker_list)+1

#         waypoint_marker = MarkerWrapper(position, **kwargs)
#         #rospy.loginfo("current pose {}".format(self.human_start_pose))
#         waypoint_marker = Marker()
#         waypoint_marker.header.frame_id = '/root' #TODO should this be root?
#         waypoint_marker.header.stamp = rospy.get_rostime()
#         waypoint_marker.ns = '/waypoint'
#         waypoint_marker.id = num_markers
#         waypoint_marker.type = Marker.SPHERE
#         waypoint_marker.pose = numpy_position_tomessage(position)
#         waypoint_marker.scale.x = self.kwargs[scale[0]
#         waypoint_marker.scale.y = scale[1]
#         waypoint_marker.scale.z = scale[2]

#         waypoint_marker.color.r = color[0]
#         waypoint_marker.color.g = color[1]
#         waypoint_marker.color.b = color[2]
#         waypoint_marker.color.a = 0.50
#         waypoint_marker.lifetime = rospy.Duration(0) 

""" A wrapper class for markers. Abstracts markers so that they can behave like a print statement
    Markers allow for visualization of positional data in rviz. Particularly useful for debug
    example usage: 
    position = np.array([0,0,0]) #create position at origin
    Markers.add_position_marker(position) #add marker to markerarray
    Markers.show() #publishes and displays markers in rviz.
"""
def show_position_marker(position, label = None, ident = 1, scale = (0.05,0.05,0.05), color = (random.random(),random.random(),random.random())):
    """ Creates and publishes a marker to rviz
        @Param label: The text label for the marker String
        @Param position: The position of the marker (3,) numpy array
        @Param color: (R,G,B) tuple (default random)
        @Param scale: (x,y,z) tuple (default 0.05)
    """

    #TODO add a repeat = false param?
    #if position in positions
    #num_markers = len(marker_list)+1

    #waypoint_marker = MarkerWrapper(position, **kwargs)
    #rospy.loginfo("current pose {}".format(self.human_start_pose))
    waypoint_marker = Marker()
    waypoint_marker.header.frame_id = '/root' #TODO should this be root?
    waypoint_marker.header.stamp = rospy.get_rostime()
    waypoint_marker.ns = '/waypoint'
    waypoint_marker.id = ident
    waypoint_marker.type = Marker.SPHERE
    waypoint_marker.pose = numpy_position_tomessage(position)
    waypoint_marker.scale.x = scale[0]
    waypoint_marker.scale.y = scale[1]
    waypoint_marker.scale.z = scale[2]

    waypoint_marker.color.r = color[0]
    waypoint_marker.color.g = color[1]
    waypoint_marker.color.b = color[2]
    waypoint_marker.color.a = 0.50
    waypoint_marker.lifetime = rospy.Duration(0)
    #if position[0] == -0.4:
       # rospy.loginfo("waypoint_marker position: {} label: {} ident: {} \nobject {}".format(position,label,ident,waypoint_marker))
    marker_pub.publish(waypoint_marker)

    if label:
        position[2] = position[2]+.01   #TODO move the z position of the text up a bit
        text_marker = Marker()
        text_marker.header.frame_id = '/root' #TOOO should this be root?
        text_marker.header.stamp = rospy.get_rostime()
        text_marker.ns = '/waypoint/text'
        text_marker.id = ident
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.pose = numpy_position_tomessage(position)
        text_marker.scale.z = 0.05
        text_marker.color.r = color[0]
        text_marker.color.g = color[1]
        text_marker.color.b = color[2]
        text_marker.color.a = 0.50
        text_marker.text = label
        text_marker.lifetime = rospy.Duration(0)
        marker_pub.publish(text_marker)

def numpy_position_tomessage(np_position):
    """ Converts a numpy (3,) position to a Point message. orientation is set to 
        @Param np_position: A (3,) numpy position
        @Return pose: A pose message set to position and orientation set to (0,0,0,1)
    """
    pose = Pose()
    pose.position = Point(np_position[0],np_position[1],np_position[2])
    pose.orientation = Quaternion(0,0,0,1)
    return pose

# def main():
#     rospy.init_node('marker_wrapper')
#     rospy.spin()

# if __name__ == '__main__':
#     try:
#         main()
#     except rospy.ROSInterruptException:
#         pass