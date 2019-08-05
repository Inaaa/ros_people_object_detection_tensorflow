#!/usr/bin/env python


import rospy
# import string
# from std_msgs.msg import String
from cob_object_detection_msgs.msg import DetectionArray
# from std_msgs.msg import Float64
from sensor_msgs.msg import Image
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import cv2
from cv_bridge import CvBridge
import message_filters
import math
import struct
import time
#import open3d as o3d

# pcl=[]
class MaskPointcloud(object):

    def __init__(self):
        super(MaskPointcloud, self).__init__()

        # init the node
        rospy.init_node('mask_point', anonymous=True)

        self._bridge = CvBridge()

        # Subscribe the rgb-image
        sub_depth = message_filters.Subscriber('/mask_depth', Image)
        sub_rgb = message_filters.Subscriber('/mask_rgb', Image)

        # Advertise the result
        self.pub = rospy.Publisher('mask_point', PointCloud2, queue_size=1)

        # Create the message filter
        ts = message_filters.ApproximateTimeSynchronizer( \
            [sub_rgb, sub_depth], \
            200, \
            20)

        ts.registerCallback(self.rgb_callback)

        self.f = 587.8
        self.cx = 351
        self.cy = 227
        self.x_res = 640
        self.y_res = 480

        # spin
        rospy.spin()

    def shutdown(self):
        """
        shuts down the node
        """
        rospy.signal_shutdown("see you later")


    def rgb_callback(self, msg, depth):
        """

        """
        time_start=time.time()
        #print(msg)
        cv_depth = self._bridge.imgmsg_to_cv2(depth, "passthrough")
        cv_msg = self._bridge.imgmsg_to_cv2(msg,"passthrough")
        time_1=time.time()
        print("time1 {}".format(time_start-time_1))

        # print("here!!!!!!!!!!!!!!!!!!!!!")
        # print(type(cv_depth))


        cv_depth_array = np.array(cv_depth)

        cv_msg_array = np.array(cv_msg)

        #print("gut!!!!!!!!!!!!!!!")
        #print(cv_msg_array)

        #print(cv_depth_array.shape)
        time_2=time.time()
        print ("time2 {}".format(time_2-time_1))




        # ros_msg=PointCloud2()
        # ros_msg.header.stamp = rospy.Time.now()
        #
        # ros_msg.height = 1
        #ros_msg.width = pcl_array.size
        time_3=time.time()
        print "time3 {}".format(time_3-time_2)
        pointcloud = []
        for v in range(cv_depth_array.shape[1]):
            for u in range(cv_depth_array.shape[0]):
                Z = cv_depth_array[u, v]
                #print(Z)
                if Z == 0.0:
                    pass
                else:
                    color = cv_msg_array[u, v]
                    X = (u-self.cx)*Z/self.f
                    Y = (v-self.cy)*Z/self.f
                    #X = (u/self.x_res - 0.5)*math.tan(self.cx*math.pi/360)*2*Z
                    #Y = (v/self.y_res - 0.5)*math.tan(self.cy*math.pi/360)*2*Z
                    #points =[X,Y,Z,0]

                    x = np.float( X )
                    y = np.float(Y)
                    z = np.float(Z)
                    #print(x,y,z)

                    r = color[0]
                    g = color[1]
                    b = color[2]
                    a = 255
                    #print(type(r))
                    #print(type(a))
                    #print(color)
                    #print (r,g,b,a)
                    # hex_r = (0xff & color[2]) << 16
                    # hex_g = (0xff & color[1]) << 8
                    # hex_b = (0xff & color[0])
                    #
                    # hex_rgb = hex_r | hex_g | hex_b
                    #
                    # float_rgb = struct.unpack('f', struct.pack('i', hex_rgb))[0]
                    time_an=time.time()
                    #float_rgb= rgb_to_float(color)
                    #points = struct.pack('ffffBBBBIII', x, y,z, 1.0, b, g, r, 0, 0, 0, 0)


                    rgb = struct.unpack('I', struct.pack('BBBB', r, g, b,a))[0]

                    #print(rgb)

                    #print hex(rgb)

                    points= [x,y,z,rgb]
                    #print(points)

                    pointcloud.append(points)
        time_zu = time.time()
        print("cicle  {}".format(time_an - time_3))
        print("generate float rgb  {}".format(time_zu - time_an))

        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('rgb', 12, PointField.UINT32, 1),
                  ]

        #             points=([np.float(X),np.float(Y),np.float(Z),color[2],color[1],color[0]])
        #             #print(points)
        #             #print("gut!!!!!!!")
        #
        #             #points=("%f %f %f %d %d %d" %(X,Y,Z,color[0],color[1],color[2]))
        #             pointcloud.append(points)

        # print pointcloud
        # print (pointcloud)
        header = Header()
        header.frame_id = "camera_rgb_optical_frame"
        #ros_msg.data = "".join(points)
        #header.frame_id = "map"
        point_generate = point_cloud2.create_cloud(header, fields, pointcloud)


        #print("successful!!!!!!!!!!!!!")

        self.pub.publish(point_generate)
        time_end=time.time()

        print("generate publish  {}".format(time_end - time_zu))
        print("generate point cloud  {}".format(time_end-time_start))

def main():
    """ main function
    """
    node = MaskPointcloud()


if __name__ == '__main__':
    main()
