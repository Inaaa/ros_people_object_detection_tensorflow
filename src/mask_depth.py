#!/usr/bin/env python


import rospy
#import string
#from std_msgs.msg import String
from cob_object_detection_msgs.msg import DetectionArray
#from std_msgs.msg import Float64
from sensor_msgs.msg import Image
#import sensor_msgs.point_cloud2 as pc2
import numpy as np
import cv2 
from cv_bridge import CvBridge
import message_filters


#pcl=[]
class MaskDepth(object):


	def __init__(self):
		super(MaskDepth,self).__init__()

		#init the node
		rospy.init_node('mask_depth', anonymous=True)

		self._bridge = CvBridge()

		#Subscribe the mask and Pointcloud
		sub_detections = message_filters.Subscriber('/object_detection/detections',DetectionArray)
		sub_depth = message_filters.Subscriber('/camera/depth_registered/sw_registered/image_rect',Image)

		#sub_pointcloud = message_filters.Subscriber('/camera/depth_registered/points',PointCloud2)

		# Advertise the result
		self.pub = rospy.Publisher('mask_depth', Image, queue_size=1)

		# Create the message filter
		ts = message_filters.ApproximateTimeSynchronizer(\
		    [sub_detections, sub_depth], \
		    200, \
		    20)

		ts.registerCallback(self.pointcloud_callback)



        	self.f = 587.8
        	self.cx = 351
        	self.cy = 227

		#spin
		rospy.spin()

	def shutdown(self):
		"""
		shuts down the node
		"""
		rospy.signal_shutdown("see you later")



	def pointcloud_callback(self,msg, depth):
		"""
	
		"""
		
		#print(depth.shape)
		cv_depth = self._bridge.imgmsg_to_cv2(depth, "passthrough")
		#cv_detections_image=self._bridge.imgmsg_to_cv2(msg,"bgr8")
		cv_depth_array=np.array(cv_depth)
		#print(cv_depth_array.shape)
		#cv_detections_image=cv2.cvtColor(detections_image,COLOR_BGR2GRAY)
		

		# get the number of detections
		no_of_detections = len(msg.detections)
		no_of_detections_type =type(no_of_detections) 
		#print(no_of_detections_type)
		#rospy.logwarn("#no_of_detections=>" + str(no_of_detections))
		#print(no_of_detections)
		#print("no_of_detections000000000000000000000000000000000000000000000000000000000")

		# Check if there is a detection of person
		

		#if False:#no_of_detections > 0:
		if no_of_detections > 0:
			for i, detection in enumerate(msg.detections):

				#print(msg.detections)
				#print(detection.header.frame_id)
				#print(detection.id)
				#print(detection.label)

				if detection.id==1:
					#print('msg.detections value{}'.format(msg.detections))
					#mask_type =type(detection.mask.mask) 
					#print(detection.mask.mask)
					#print(mask_type)
					cv_detections_mask=self._bridge.imgmsg_to_cv2(detection.mask.mask,"passthrough")
					cv_detections_mask_array=np.array(cv_detections_mask)

					#print(cv_detections_mask_array.shape)
					#print(cv_detections_mask_array)





					##width =  detection.mask.roi.width
					##height = detection.mask.roi.height
					#cv_depth_resize=cv2.resize(cv_depth,(width,height),interpolation=cv2.INTER_AREA)
					#cv_mask_resize=cv2.resize(cv_detections_mask,(width,height),interpolation=cv2.INTER_AREA)




					

					#msg=cv2.bitwise_and(np.array(cv_depth), np.array(cv_depth), mask=np.array(cv_detections_image))

					#msg=cv2.bitwise_and(np.array(cv_depth_resize), np.array(cv_depth_resize), mask=np.array(cv_mask_resize))
					msg=cv2.bitwise_and(np.array(cv_depth), np.array(cv_depth), mask=np.array(cv_detections_mask))
					msg_im=self._bridge.cv2_to_imgmsg(msg, encoding="passthrough")
					#print("here is person!!!!!!!!!!!!!!!!!!!!!!!")

					self.pub.publish(msg_im)
					break



def main():
	""" main function
	"""
	node = MaskDepth()

if __name__ == '__main__':
	main()

