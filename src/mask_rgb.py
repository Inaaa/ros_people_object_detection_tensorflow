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
class MaskPointcloud(object):


	def __init__(self):
		super(MaskPointcloud,self).__init__()

		#init the node
		rospy.init_node('mask_rgb', anonymous=True)

		self._bridge = CvBridge()

		#Subscribe the rgb-image
		sub_detections = message_filters.Subscriber('/object_detection/detections',DetectionArray)		
		sub_rgb = message_filters.Subscriber('/camera/rgb/image_rect_color',Image)
	

		# Advertise the result
		self.pub = rospy.Publisher('mask_rgb', Image, queue_size=1)

		# Create the message filter
		ts = message_filters.ApproximateTimeSynchronizer(\
		    [sub_detections, sub_rgb], \
		    200, \
		    20)

		ts.registerCallback(self.rgb_callback)



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



	def rgb_callback(self,msg, depth):
		"""
	
		"""
		
		
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
		#print("rgb__no_of_detections000000000000000000000000000000000000000000000000000000000")

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

					#msg=cv2.bitwise_and(np.array(cv_depth), np.array(cv_depth), mask=np.array(cv_detections_image))

					#msg=cv2.bitwise_and(np.array(cv_depth_resize), np.array(cv_depth_resize), mask=np.array(cv_mask_resize))
					msg=cv2.bitwise_and(np.array(cv_depth), np.array(cv_depth), mask=np.array(cv_detections_mask))
					msg_im=self._bridge.cv2_to_imgmsg(msg, encoding="rgb8")
					#msg_im=self._bridge.cv2_to_imgmsg(msg, encoding="bgr8")
					msg_array=np.array(msg)
					#print(msg_array.shape)
					#print("here is person!!!!!!!!!!!!!!!!!!!!!!!")

					self.pub.publish(msg_im)


					
		



		#for p in pc2.read_points(depth, field_names = ("x", "y", "z"), skip_nans=True):
         		#pcl.append([p[0],p[1],p[2]])

		#result=cv2.bitwise_and(np.array(depth), np.array(depth), mask=msg.detections)
		#print('pointcloud value{}'.format(self.pointcloud))
		
			



#print " x : %f  y: %f  z: %f" %(p[0],p[1],p[2])
			#print(pcl)


      #pass
      
      #print('pointcloud value{}'.format(self.pointcloud))
      #result=cv2.bitwise_and(np.asarray(pcl), np.asarray(self.pointcloud), mask=self.detections)
      #result=cv2.bitwise_and(np.array(pcl), np.array(pcl), mask=self.detections)
      
 #     print self.pointcloud
    
#if __name__ == '__main__':
  
#	node = MaskPointcloud()
#	rospy.spin()

def main():
	""" main function
	"""
	node = MaskPointcloud()

if __name__ == '__main__':
	main()

