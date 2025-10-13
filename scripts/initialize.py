#!/usr/bin/env python3

import rclpy
import ros2_numpy
from trackdlo.trackdlo_parameters_py import trackdlo
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.node import Node
import rclpy.time
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
import sensor_msgs_py.point_cloud2 as pcl2
import std_msgs.msg
import message_filters

import struct
import time
import cv2
import numpy as np

from visualization_msgs.msg import MarkerArray
from scipy import interpolate

from utils import extract_connected_skeleton, ndarray2MarkerArray

from sensor_msgs.msg import PointField
import numpy as np


class TrackerInitializer(Node):

    def __init__(self):
        super().__init__('init_tracker')
        self.proj_matrix = None

        param_listener = trackdlo.ParamListener(self)
        params = param_listener.get_params()

        self.num_of_nodes = params.num_of_nodes
        self.multi_color_dlo =  params.multi_color_dlo
        self.camera_info_topic = params.camera_info_topic
        self.rgb_topic = params.rgb_topic
        self.depth_topic = params.depth_topic
        self.result_frame_id = params.result_frame_id
        self.visualize_initialization_process = params.visualize_initialization_process
        self.hsv_threshold_upper_limit = params.hsv_threshold_upper_limit
        self.hsv_threshold_lower_limit = params.hsv_threshold_lower_limit

        upper_array = self.hsv_threshold_upper_limit.split(' ')
        lower_array = self.hsv_threshold_lower_limit.split(' ')
        self.upper = (int(upper_array[0]), int(upper_array[1]), int(upper_array[2]))
        self.lower = (int(lower_array[0]), int(lower_array[1]), int(lower_array[2]))

        self.camera_info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
        rclpy.spin_once(self)

        self.rgb_sub = message_filters.Subscriber(self, Image, self.rgb_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic)

        # header
        self.header = std_msgs.msg.Header()
        self.header.stamp = self.get_clock().now().to_msg()
        self.header.frame_id = self.result_frame_id

        self.fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                    PointField(name='rgba', offset=12, datatype=PointField.UINT32, count=1)]

        self.pc_pub = self.create_publisher(PointCloud2, '/trackdlo/init_nodes', 10)
        self.results_pub = self.create_publisher(MarkerArray, '/trackdlo/init_nodes_markers', 10) 

        self.bridge = CvBridge()

        ts = message_filters.TimeSynchronizer([self.rgb_sub, self.depth_sub], 10)
        ts.registerCallback(self.callback)
        

    def camera_info_callback(self, info):
        self.proj_matrix = np.array(list(info.p)).reshape(3, 4)
        print('Received camera projection matrix:')
        print(self.proj_matrix)
        self.destroy_subscription(self.camera_info_sub) 

    def color_thresholding(self, hsv_image, cur_depth):
        mask_dlo = cv2.inRange(hsv_image.copy(), self.lower, self.upper).astype('uint8')

        # tape green
        lower_green = (58, 130, 50)
        upper_green = (90, 255, 89)
        mask_green = cv2.inRange(hsv_image.copy(), lower_green, upper_green).astype('uint8')

        # combine masks
        mask = cv2.bitwise_or(mask_green.copy(), mask_dlo.copy())

        # filter mask base on depth values
        mask[cur_depth < 0.57*1000] = 0

        return mask, mask_green

    def remove_duplicate_rows(self, array):
        _, idx = np.unique(array, axis=0, return_index=True)
        data = array[np.sort(idx)]
        #self.get_parameter('num_of_nodes')
        return data

    def callback(self, rgb, depth):
        #print("Initializing...")
        init_time = time.time()
        # process rgb image
        # cur_image = ros_numpy.numpify(rgb) # ORIGINAL
        # hsv_image = cv2.cvtColor(cur_image.copy(), cv2.COLOR_RGB2HSV) # ORIGINAL

        #cur_image = from_ros_msg(rgb) 
        #cur_image = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')

        cur_image = ros2_numpy.numpify(rgb)  
        #cur_image = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='rgb8')
        hsv_image = cv2.cvtColor(cur_image.copy(), cv2.COLOR_RGB2HSV)

        #cv2.imshow('Initial image', cur_image)
        #cv2.waitKey(0)

        # process depth image
        cur_depth = ros2_numpy.numpify(depth)
        #cur_depth = self.bridge.imgmsg_to_cv2(depth, desired_encoding='passthrough')
        #cur_depth = np.array(cur_depth, dtype=np.float32)
        #cur_depth = cur_depth.astype(np.uint16)

        if not self.multi_color_dlo:
            # color thresholding
            mask = cv2.inRange(hsv_image, self.lower, self.upper)
            #cv2.imshow('mask image', mask)
            #cv2.waitKey(0)
        else:
            # color thresholding
            mask, mask_tip = self.color_thresholding(hsv_image, cur_depth)

        try:
            start_time = time.time()
            mask = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)

            # returns the pixel coord of points (in order). a list of lists
            img_scale = 1
            extracted_chains = extract_connected_skeleton(self.visualize_initialization_process, mask, img_scale=img_scale, seg_length=8, max_curvature=25)
            #print('num of chains (django): ', extracted_chains)

            all_pixel_coords = []
            for chain in extracted_chains:
                all_pixel_coords += chain
            #print('Finished extracting chains. Time taken:', time.time()-start_time)

            all_pixel_coords = np.array(all_pixel_coords) * img_scale
            all_pixel_coords = np.flip(all_pixel_coords, 1)

            pc_z = cur_depth[tuple(map(tuple, all_pixel_coords.T))] / 1000.0
            fx = self.proj_matrix[0, 0]
            fy = self.proj_matrix[1, 1]
            cx = self.proj_matrix[0, 2]
            cy = self.proj_matrix[1, 2]
            pixel_x = all_pixel_coords[:, 1]
            pixel_y = all_pixel_coords[:, 0]
            # if the first mask value is not in the tip mask, reverse the pixel order
            if self.multi_color_dlo:
                pixel_value1 = mask_tip[pixel_y[-1],pixel_x[-1]]
                if pixel_value1 == 255:
                    pixel_x, pixel_y = pixel_x[::-1], pixel_y[::-1]

            pc_x = (pixel_x - cx) * pc_z / fx
            pc_y = (pixel_y - cy) * pc_z / fy
            extracted_chains_3d = np.vstack((pc_x, pc_y))
            extracted_chains_3d = np.vstack((extracted_chains_3d, pc_z))
            extracted_chains_3d = extracted_chains_3d.T

            # do not include those without depth values
            extracted_chains_3d = extracted_chains_3d[((extracted_chains_3d[:, 0] != 0) | (extracted_chains_3d[:, 1] != 0) | (extracted_chains_3d[:, 2] != 0))]

            if self.multi_color_dlo:
                depth_threshold = 0.57  # m
                extracted_chains_3d = extracted_chains_3d[extracted_chains_3d[:, 2] > depth_threshold]

            # tck, u = interpolate.splprep(extracted_chains_3d.T, s=0.001)
            tck, u = interpolate.splprep(extracted_chains_3d.T, s=0.0005)
            # 1st fit, less points
            u_fine = np.linspace(0, 1, 300) # <-- num fit points
            x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
            spline_pts = np.vstack((x_fine, y_fine, z_fine)).T

            # 2nd fit, higher accuracy
            num_true_pts = int(np.sum(np.sqrt(np.sum(np.square(np.diff(spline_pts, axis=0)), axis=1))) * 1000)
            u_fine = np.linspace(0, 1, num_true_pts) # <-- num true points
            x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
            spline_pts = np.vstack((x_fine, y_fine, z_fine)).T
            nodes = spline_pts[np.linspace(0, num_true_pts-1, self.num_of_nodes).astype(int)]

            init_nodes = self.remove_duplicate_rows(nodes)
            #results = ndarray2MarkerArray(init_nodes, self.result_frame_id, [1, 150/255, 0, 0.75], [0, 1, 0, 0.75])
            #print(init_nodes)
            results = ndarray2MarkerArray(init_nodes, self.result_frame_id, [0.0, 149/255, 203/255, 0.75], [0.0, 149/255, 203/255, 0.75])
            self.results_pub.publish(results)

            # add color
            pc_rgba = struct.unpack('I', struct.pack('BBBB', 255, 40, 40, 255))[0]
            pc_rgba_arr = np.full((len(init_nodes), 1), pc_rgba)
            pc_colored = np.hstack((init_nodes, pc_rgba_arr)).astype(object)
            pc_colored[:, 3] = pc_colored[:, 3].astype(int)

            dtype = np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgba', np.uint32)
            ])
            pc_colored_structured = np.empty(init_nodes.shape[0], dtype=dtype)
            pc_colored_structured['x'] = init_nodes[:, 0]
            pc_colored_structured['y'] = init_nodes[:, 1]
            pc_colored_structured['z'] = init_nodes[:, 2]
            pc_colored_structured['rgba'] = pc_rgba_arr.flatten() # Ensure it's 1D

            self.header.stamp = self.get_clock().now().to_msg()
            converted_points = pcl2.create_cloud(self.header, self.fields, pc_colored_structured)
            self.pc_pub.publish(converted_points)
            #print('Done with callback, Time taken:', time.time()-init_time)
        except Exception as e:
            self.get_logger().error(e)
            self.get_logger().error("Failed to extract splines.")
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    tracker_initializer = TrackerInitializer()
    rclpy.spin(tracker_initializer)

    tracker_initializer.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()
    
