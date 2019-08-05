#!/usr/bin/env python

"""
A ROS node to generate pointcloud of people via depth image ,rgb image and result of Mask_Rcnn.

Author:
	Chanchan Li -- chanchan.li@ipa.fraunhofer.de

"""




import rospy

import numpy as np

import cv2

from cv_bridge import CvBridge

import message_filters

import math

import struct

import time

from cob_object_detection_msgs.msg import DetectionArray

from sensor_msgs import point_cloud2

from sensor_msgs.msg import Image ,PointCloud2, PointField

from std_msgs.msg import Header

import sensor_msgs.point_cloud2 as pc2



class MaskPointcloud(object):

    def __init__(self):
        super(MaskPointcloud, self).__init__()

        # init the node
        rospy.init_node('mask_pointcloud', anonymous=True)

        self._bridge = CvBridge()

        # Subscribe the mask_image, depth_image und rgb_image
        sub_detections = message_filters.Subscriber('/object_detection/detections', DetectionArray)
        sub_depth = message_filters.Subscriber('/camera/depth_registered/sw_registered/image_rect', Image)
        sub_rgb = message_filters.Subscriber('/camera/rgb/image_rect_color', Image)


        # Advertise the result
        self.pub = rospy.Publisher('mask_pointcloud', PointCloud2, queue_size=2)

        # Create the message filter
        ts = message_filters.ApproximateTimeSynchronizer( \
            [sub_detections, sub_depth,sub_rgb], \
            20, \
            5)

        ts.registerCallback(self.pointcloud_callback)

        self.f = 587.8
        self.cx = 351
        self.cy = 227

        # spin
        rospy.spin()

    def shutdown(self):
        """
        shuts down the node
        """
        rospy.signal_shutdown("see you later")

    def pointcloud_callback(self, msg, depth, rgb):
        """
	msg:
	   object detection massage
	depth:
	   depth image of camera
	rgb:
	   rgb image of camera

        """
        #time_start=time.time()
	
	#convert image to numpy array
        cv_depth = self._bridge.imgmsg_to_cv2(depth, "passthrough")
        cv_rgb = self._bridge.imgmsg_to_cv2(rgb, "passthrough")


        # get the number of detections
        no_of_detections = len(msg.detections)

       
        if no_of_detections > 0:
            for i, detection in enumerate(msg.detections):

		#check detection is a people und one people in a image
                if detection.id == 1:

                    #convert detection_image to numpy array
                    cv_detections_mask = self._bridge.imgmsg_to_cv2(detection.mask.mask, "passthrough")
                    cv_detections_mask_array = np.array(cv_detections_mask)
		    
		    #mask depth image through detection image
                    msg_depth = cv2.bitwise_and(np.array(cv_depth), np.array(cv_depth), mask=np.array(cv_detections_mask))
                    
                    #mask rgb image through detection_image 
                    msg_rgb = cv2.bitwise_and(np.array(cv_rgb), np.array(cv_rgb), mask=np.array(cv_detections_mask))


                    #finden the box
                    x_box = detection.mask.roi.x
                    y_box = detection.mask.roi.y
                    width = detection.mask.roi.width
                    height = detection.mask.roi.height
                    print ("x_box= {}".format(x_box))
                    print ("y_box= {}".format(y_box))
                    print ("width {}".format(width))
                    print ("height= {}".format(height))

                    break

	#generate pointloud
        pcl=self.generatepointcloud(msg_rgb, msg_depth, x_box, y_box, width, height)

	#publish the massage
        self.pub.publish(pcl)

        #time_end = time.time()
        #print("new point generate Time {}".format(time_end-time_start))



    def generatepointcloud(self, msg_rgb, msg_depth,x_box,y_box,width,height):
        """
	generate pointcloud using mask_rgb image und msg_depth image
	
	msg_rgb:
	   mask_rgb image
	msg_depth:
	   mask_depth image
	x_box:
	   x coordinate of people box
	y_box:
	   y coordinate of people box
	width:
	   the width of people box
	height:
	   the height of people box
	
        """

        msg_rgb_array = np.array(msg_rgb)
        msg_depth_array = np.array(msg_depth)
        #print(msg_depth_array.shape)

        pointcloud = []
        for u in range(width):
            for v in range(height):
                v_real = v+y_box
                u_real = u+x_box
                Z = msg_depth_array[v_real, u_real]
                #print(Z)
                if Z == 0:
                    pass
                else:
                    color = msg_rgb_array[v_real, u_real]
                    X = (u_real-self.cx)*Z/self.f
                    Y = (v_real-self.cy)*Z/self.f
                    x = np.float( X )
                    y = np.float(Y)
                    z = np.float(Z)
                    #print(x,y,z)

                    r = color[2]
                    g = color[1]
                    b = color[0]
                    a = 255

                    rgb = struct.unpack('I', struct.pack('BBBB', r, g, b, a))[0]

                    points= [x,y,z,rgb]

                    pointcloud.append(points)

        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('rgb', 12, PointField.UINT32, 1),
                  ]
        
    
        header =Header()
        header.frame_id = "camera_rgb_optical_frame"

        point_generate = point_cloud2.create_cloud(header, fields, pointcloud)
        point_generate.header.stamp = rospy.Time.now()
        #print("successful!!!!!!!!!!!!!")
        return point_generate


def main():
    """ main function
    """
    node = MaskPointcloud()


if __name__ == '__main__':
    main()
