#!/usr/bin/env python3



# Team ID:          [ CL#1906]
# Author List:		[ Koushik S,M.Girish Raghav,Allen Joseph,K.Manivel ]
# Filename:		    task1a.py
# Functions:
#			        [ Comma separated list of functions in this file ]
# Nodes:		    Add your publishing and subscribing node
#                   Example:
#			        Publishing Topics  - [ /tf ]
#                   Subscribing Topics - [ /camera/aligned_depth_to_color/image_raw, /etc... ]


################### IMPORT MODULES ##########

import rclpy    
import sys
import cv2
import math
from sensor_msgs.msg import PointCloud2
import tf2_ros
from tf2_ros import TransformException
from rclpy.time import Time
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image


##################### FUNCTION DEFINITIONS #######################

def calculate_rectangle_area(coordinates):
    
    #linalg.norm calculates the length of the vector in this case the width and height
    width = np.linalg.norm(coordinates[0] - coordinates[1])
    height = np.linalg.norm(coordinates[1] - coordinates[2])
    
    area = width * height

    return area, width

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x,y,z])
    
def detect_aruco(image):

    if image is None or image.size == 0:
        print("Error: Empty or invalid image.")
        return [], [], [], [], []
    
    #image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    # Creating a 4x4 ArUco dictionary
    parameters = cv2.aruco.DetectorParameters()  
    # Detect ArUco markers.
    corners, ids, _ = cv2.aruco.detectMarkers(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), aruco_dict, parameters=parameters)
    
    cam_mat = np.array([[931.1829833984375, 0.0, 640.0], [0.0, 931.1829833984375, 360.0], [0.0, 0.0, 1.0]],dtype=np.float64)
    dist_mat = np.array([0.0, 0.0, 0.0, 0.0, 0.0],dtype=np.float64)
    cv2.aruco.drawDetectedMarkers(image, corners, ids)  
    # Estimate the pose of each ArUco marker.
    center_aruco_list = []
    distance_from_rgb_list = []
    angle_aruco_list = []
    width_aruco_list = []
    r_ids = []

    if ids is not None:
        for i in range(len(ids)):
            corner = corners[i][0].astype(int)
            area, _ = calculate_rectangle_area(corner)
            cX = int(np.mean(corner[:, 0]))
            cY = int(np.mean(corner[:, 1]))
            center_aruco_list.append([cX, cY])
            
            
            if area > 1500:
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.15, cam_mat,dist_mat)
                rvec = rvec[i]  
                tvec = tvec[i]  
                
                distance_from_rgb = np.linalg.norm(tvec)
                distance_from_rgb_list.append(distance_from_rgb)
                
                # Draw axes on the image
                cv2.drawFrameAxes(image, cam_mat, dist_mat, rvec, tvec, 1.0)

                # Draw markers and IDs on the image
                
                rmat, _ = cv2.Rodrigues(rvec)

            # Convert the rotation matrix to Euler angles
                _, _, yaw = rotationMatrixToEulerAngles(rmat)
                
                angle_aruco_list.append(yaw)

                # Calculate the width of the ArUco marker.
                width_aruco = corner[2][0] - corner[0][0]
                width_aruco_list.append(width_aruco)

                # Save the ArUco marker ID.
                r_ids.append(ids[i][0])
    return center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_aruco_list, r_ids


class aruco_tf(Node):

    def __init__(self):

        super().__init__('aruco_tf_publisher')                                          # registering node

        ############ Topic SUBSCRIPTIONS ############

        self.color_cam_sub = self.create_subscription(Image, '/camera/color/image_raw', self.colorimagecb, 10)
        self.depth_cam_sub = self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.depthimagecb, 10)
        self.pointcloud_sub = self.create_subscription(PointCloud2, '/camera/depth/color/points', self.pointcloud_callback, 10)

        ############ Constructor VARIABLES/OBJECTS ############

        image_processing_rate = 0.2                                                     # rate of time to process image (seconds)
        self.bridge = CvBridge()                                                        # initialise CvBridge object for image conversion
        self.tf_buffer = tf2_ros.buffer.Buffer()                                        # buffer time used for listening transforms
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)                                    # object as transform broadcaster to send transform wrt some frame_id
        self.timer = self.create_timer(image_processing_rate, self.process_image)       # creating a timer based function which gets called on every 0.2 seconds (as defined by 'image_processing_rate' variable)

        self.cv_image = None                                                            # colour raw image variable (from colorimagecb())
        self.depth_image = None                                                         # depth image variable (from depthimagecb())
    def pointcloud_callback(self,msg):
        pass

    def depthimagecb(self, data):
        try:
            self.depth_image=self.bridge.imgmsg_to_cv2(data, "passthrough")
        except CvBridgeError as e:
            print(e)


    def colorimagecb(self, data):
       
        self.cv_image=self.bridge.imgmsg_to_cv2(data, "passthrough")
        cv2.flip(self.cv_image,-1);
        cv2.rotate(self.cv_image, cv2.ROTATE_90_CLOCKWISE)

       
    def euler_to_quaternion(self, roll, pitch, yaw):
        # Create a Rotation object from the Euler angles
        rotation = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
        # Convert the Rotation object to a quaternion
        quat = rotation.as_quat()
        return quat
    
    
    def publish_transform(self, quat, trt, marker_id):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'camera_link'
        t.child_frame_id = f'cam_{marker_id}'
        t.transform.translation.x = trt[0]
        t.transform.translation.y = trt[1]
        t.transform.translation.z = trt[2]
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        self.br.sendTransform(t)

    def lookup_object_transform(self, marker_id):
        try:
            base_to_obj_transform = self.tf_buffer.lookup_transform('base_link', f'shoulder_link', Time())
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'base_link'
            t.child_frame_id = f'obj_{marker_id}'
            t.transform.translation.x = base_to_obj_transform.transform.translation.x
            t.transform.translation.y = base_to_obj_transform.transform.translation.y
            t.transform.translation.z = base_to_obj_transform.transform.translation.z
            t.transform.rotation.x = base_to_obj_transform.transform.rotation.x
            t.transform.rotation.y = base_to_obj_transform.transform.rotation.y
            t.transform.rotation.z = base_to_obj_transform.transform.rotation.z
            t.transform.rotation.w = base_to_obj_transform.transform.rotation.w
            self.br.sendTransform(t)
        except tf2_ros.LookupException as e:
            self.get_logger().info(f'Could not transform base_link to obj_{str(marker_id)}: {str(e)}')

    def calculate_realsense_depth(self,center_aruco_list, depth_image):
        depths = []
        for center in center_aruco_list:
            x_depth = int(center[0])
            y_depth = int(center[1])
            depth = depth_image[y_depth, x_depth]  # Depth in millimeters
            depths.append(depth / 1000.0)  # Convert mm to meters
        return depths
    def process_image(self):
        if self.cv_image is None:
            print("Color image not available yet.")
            return

        sizeCamX = 1280
        sizeCamY = 720
        centerCamX = 640 
        centerCamY = 360
        focalX = 931.1829833984375
        focalY = 931.1829833984375

        center_aruco_list, distance_from_rgb_list, angle_aruco_list, _, ids = detect_aruco(self.cv_image)
        realsense_depths = self.calculate_realsense_depth(center_aruco_list, self.depth_image)
        for i in range(len(ids)):
            # Correcting aruco angle
            angle_aruco_list[i] = (0.788 * angle_aruco_list[i]) - ((angle_aruco_list[i] ** 2) / 3160)
            # Getting quaternions from roll pitch yaw
            euler_angles = [0, 0, angle_aruco_list[i]]
            quat = self.euler_to_quaternion(euler_angles[0], euler_angles[1], euler_angles[2])
            msg = PointCloud2()
            # Use the correct x, y, and z for translation
            x_trans = realsense_depths[i] * (sizeCamX - center_aruco_list[i][0] - centerCamX) / focalX
            y_trans = realsense_depths[i] * (sizeCamY - center_aruco_list[i][1] - centerCamY) / focalY
            z_trans = realsense_depths[i]
            trt = [x_trans, y_trans, z_trans]
            for j in range(min(len(center_aruco_list),len(ids))):
                cv2.circle(self.cv_image, (int(center_aruco_list[j][0]), int(center_aruco_list[j][1])), 5, (0, 0, 255), -1)
                cv2.putText(self.cv_image, str(ids[j]), (int(center_aruco_list[j][0]), int(center_aruco_list[j][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            self.publish_transform(quat, trt, ids[i])
            self.lookup_object_transform( ids[i])
            
        # Showing image
        cv2.imshow("Color Image", self.cv_image)
        cv2.waitKey(1)


def main():
    

    rclpy.init(args=sys.argv)                                       # initialisation

    node = rclpy.create_node('aruco_tf_process')                    # creating ROS node

    node.get_logger().info('Node created: Aruco tf process')        # logging information

    aruco_tf_class = aruco_tf()                                     # creating a new object for class 'aruco_tf'

    rclpy.spin(aruco_tf_class)                                      # spining on the object to make it alive in ROS 2 DDS

    aruco_tf_class.destroy_node()                                   # destroy node after spin ends

    rclpy.shutdown()                                                # shutdown process


if __name__ == '__main__':
    main()
