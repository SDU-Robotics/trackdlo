#!/usr/bin/env python3

import struct
from collections import deque

import cv2
import message_filters
import numpy as np
import rclpy
import rclpy.time
import sensor_msgs_py.point_cloud2 as pcl2
import std_msgs.msg
from cv_bridge import CvBridge
from dlo_segmentation import DloSegmenter
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from trackdlo.trackdlo_parameters_py import trackdlo
from utils import extract_connected_skeleton, ndarray2MarkerArray
from visualization_msgs.msg import MarkerArray

import ros2_numpy

DEFAULT_CHECKPOINT = '/home/madelocal/trackdlo_ros2_ws/src/trackdlo/segmentation_model' + "/multiview_tracking_dlo_segmentation_model.pth"

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
        self.enable_consensus_depth_correction = getattr(params, 'enable_consensus_depth_correction', False)
        self.init_nodes_published = False

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
        self.mask_pub = self.create_publisher(Image, '/mask', 10)
        self.init_debug_img_pub = self.create_publisher(Image, '/trackdlo/init_debug_img', 10)

        self.bridge = CvBridge()
        self.segmenter = DloSegmenter(DEFAULT_CHECKPOINT, device='cpu')

        ts = message_filters.TimeSynchronizer([self.rgb_sub, self.depth_sub], 10)
        ts.registerCallback(self.callback)
        

    def camera_info_callback(self, info):
        self.proj_matrix = np.array(list(info.p)).reshape(3, 4)
        print('Received camera projection matrix:')
        print(self.proj_matrix)
        self.destroy_subscription(self.camera_info_sub) 

    def remove_duplicate_rows(self, array):
        _, idx = np.unique(array, axis=0, return_index=True)
        data = array[np.sort(idx)]
        #self.get_parameter('num_of_nodes')
        return data

    def merge_extracted_chains(self, chains, max_bridge_gap_px=220.0):
        proc = []
        for chain in chains:
            arr = np.asarray(chain, dtype=np.int32)
            if arr.ndim == 2 and arr.shape[0] >= 2:
                proc.append(arr)

        if not proc:
            return np.empty((0, 2), dtype=np.int32)

        # Start from the longest chain and greedily attach others by nearest endpoints.
        lengths = [float(np.sum(np.linalg.norm(np.diff(ch, axis=0), axis=1))) for ch in proc]
        seed_idx = int(np.argmax(lengths))
        merged = proc.pop(seed_idx)

        while proc:
            best = None
            for idx, ch in enumerate(proc):
                ch_start = ch[0]
                ch_end = ch[-1]
                m_start = merged[0]
                m_end = merged[-1]

                candidates = [
                    (np.linalg.norm(m_end - ch_start), 'append_forward'),
                    (np.linalg.norm(m_end - ch_end), 'append_reverse'),
                    (np.linalg.norm(m_start - ch_end), 'prepend_forward'),
                    (np.linalg.norm(m_start - ch_start), 'prepend_reverse'),
                ]
                dist, mode = min(candidates, key=lambda t: t[0])
                if best is None or dist < best[0]:
                    best = (dist, idx, mode)

            if best is None:
                break

            dist, best_idx, mode = best
            if dist > max_bridge_gap_px:
                break

            ch = proc.pop(best_idx)
            if mode == 'append_forward':
                merged = np.vstack((merged, ch))
            elif mode == 'append_reverse':
                merged = np.vstack((merged, ch[::-1]))
            elif mode == 'prepend_forward':
                merged = np.vstack((ch, merged))
            else:  # prepend_reverse
                merged = np.vstack((ch[::-1], merged))

        # Remove duplicate consecutive points.
        keep = [0]
        for i in range(1, merged.shape[0]):
            if not np.array_equal(merged[i], merged[i - 1]):
                keep.append(i)
        merged = merged[np.asarray(keep, dtype=np.int32)]

        if merged.shape[0] < 2:
            return np.empty((0, 2), dtype=np.int32)

        # extract_connected_skeleton returns (x, y); convert to (row, col).
        merged_rc = np.column_stack((merged[:, 1], merged[:, 0])).astype(np.int32)
        return merged_rc

    def sample_polyline_pixels_by_arclength(self, polyline_rc, num_samples):
        polyline_rc = np.asarray(polyline_rc, dtype=np.float64)
        if polyline_rc.ndim != 2 or polyline_rc.shape[0] < num_samples:
            return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.int32)

        seg_len = np.linalg.norm(np.diff(polyline_rc, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(seg_len)))
        if cumulative[-1] <= 1e-9:
            return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.int32)

        targets = np.linspace(0.0, cumulative[-1], num_samples)
        sampled_idx = []
        for target in targets:
            right = int(np.searchsorted(cumulative, target, side='left'))
            right = min(max(right, 0), len(cumulative) - 1)
            left = max(right - 1, 0)

            if right == left:
                idx = right
            elif abs(cumulative[right] - target) < abs(cumulative[left] - target):
                idx = right
            else:
                idx = left

            if sampled_idx and idx <= sampled_idx[-1]:
                idx = min(sampled_idx[-1] + 1, polyline_rc.shape[0] - 1)

            sampled_idx.append(idx)

        sampled_idx = np.asarray(sampled_idx, dtype=np.int32)
        sampled_pixels = np.rint(polyline_rc[sampled_idx]).astype(np.int32)
        return sampled_pixels, sampled_idx

    def consensus_correct_node_depths(self, node_depths):
        node_depths = np.asarray(node_depths, dtype=np.float64)
        if node_depths.ndim != 1 or node_depths.size < 5:
            return node_depths, 0

        corrected = node_depths.copy()
        replaced = 0
        half_window = 2

        for i in range(corrected.size):
            start = max(0, i - half_window)
            end = min(corrected.size, i + half_window + 1)
            neighborhood = corrected[start:end]
            if neighborhood.size <= 1:
                continue

            center_offset = i - start
            neighbors_only = np.delete(neighborhood, center_offset)
            if neighbors_only.size < 2:
                continue

            local_median = float(np.median(neighbors_only))
            local_mad = float(np.median(np.abs(neighbors_only - local_median)))
            vote_threshold = max(0.02, 2.5 * local_mad)

            inlier_votes = int(np.count_nonzero(np.abs(neighbors_only - corrected[i]) <= vote_threshold))
            min_votes = int(np.ceil(0.6 * neighbors_only.size))

            if inlier_votes < min_votes:
                corrected[i] = local_median
                replaced += 1

        return corrected, replaced

    def centerline_pixels_to_3d_nodes(self, centerline_rc, sampled_idx, cur_depth):
        centerline_rc = np.asarray(centerline_rc, dtype=np.int32)
        sampled_idx = np.asarray(sampled_idx, dtype=np.int32)
        if centerline_rc.ndim != 2 or sampled_idx.ndim != 1 or sampled_idx.size == 0:
            return np.empty((0, 3), dtype=np.float64)

        pixel_y = np.clip(centerline_rc[:, 0], 0, cur_depth.shape[0] - 1)
        pixel_x = np.clip(centerline_rc[:, 1], 0, cur_depth.shape[1] - 1)
        all_depth = cur_depth[pixel_y, pixel_x].astype(np.float64) / 1000.0

        valid_depth = np.isfinite(all_depth) & (all_depth > 0.05) & (all_depth < 3.0)
        if self.multi_color_dlo:
            valid_depth = valid_depth & (all_depth > 0.57)

        valid_idx = np.where(valid_depth)[0]
        if valid_idx.size < 2:
            return np.empty((0, 3), dtype=np.float64)

        # Interpolate depth along ordered centerline so sparse holes do not collapse nodes to one side.
        interp_depth = np.interp(np.arange(all_depth.shape[0], dtype=np.float64), valid_idx.astype(np.float64), all_depth[valid_idx])

        node_rows = np.clip(centerline_rc[sampled_idx, 0], 0, cur_depth.shape[0] - 1)
        node_cols = np.clip(centerline_rc[sampled_idx, 1], 0, cur_depth.shape[1] - 1)
        node_depth = interp_depth[sampled_idx]
        if self.enable_consensus_depth_correction:
            node_depth, replaced_count = self.consensus_correct_node_depths(node_depth)
            if replaced_count > 0:
                self.get_logger().warn(
                    f'Consensus depth correction replaced {replaced_count} outlier init node depth(s).'
                )

        fx = self.proj_matrix[0, 0]
        fy = self.proj_matrix[1, 1]
        cx = self.proj_matrix[0, 2]
        cy = self.proj_matrix[1, 2]

        pc_x = (node_cols.astype(np.float64) - cx) * node_depth / fx
        pc_y = (node_rows.astype(np.float64) - cy) * node_depth / fy
        points_3d = np.column_stack((pc_x, pc_y, node_depth))

        points_3d = points_3d[np.isfinite(points_3d).all(axis=1)]
        if points_3d.shape[0] < self.num_of_nodes:
            return np.empty((0, 3), dtype=np.float64)

        return points_3d

    def publish_init_debug_image(self, rgb_msg, cur_image, mask_mono, node_pixels_rc):
        debug_img = cur_image.copy()
        if debug_img.ndim == 2:
            debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)

        mask_edges, _ = cv2.findContours(mask_mono, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(debug_img, mask_edges, -1, (255, 0, 0), 1)

        for rc in node_pixels_rc:
            r = int(rc[0])
            c = int(rc[1])
            cv2.circle(debug_img, (c, r), 3, (0, 255, 255), -1)

        debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
        debug_msg.header = rgb_msg.header
        self.init_debug_img_pub.publish(debug_msg)

    def sample_curve_by_arclength(self, points_3d, num_samples):
        points_3d = np.asarray(points_3d, dtype=np.float64)
        if points_3d.ndim != 2 or points_3d.shape[0] < num_samples:
            return np.empty((0, 3), dtype=np.float64)

        segment_lengths = np.linalg.norm(np.diff(points_3d, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        if cumulative[-1] <= 1e-9:
            return np.empty((0, 3), dtype=np.float64)

        targets = np.linspace(0.0, cumulative[-1], num_samples)
        sampled_idx = []
        for target in targets:
            right = int(np.searchsorted(cumulative, target, side='left'))
            right = min(max(right, 0), len(cumulative) - 1)
            left = max(right - 1, 0)

            if right == left:
                idx = right
            elif abs(cumulative[right] - target) < abs(cumulative[left] - target):
                idx = right
            else:
                idx = left

            if sampled_idx and idx <= sampled_idx[-1]:
                idx = min(sampled_idx[-1] + 1, points_3d.shape[0] - 1)

            sampled_idx.append(idx)

        sampled_idx = np.asarray(sampled_idx, dtype=np.int32)
        sampled_points = points_3d[sampled_idx]
        return self.remove_duplicate_rows(sampled_points)

    def callback(self, rgb, depth):
        # process rgb image
        cur_image = ros2_numpy.numpify(rgb)  

        #cv2.imshow('Initial image', cur_image)
        #cv2.waitKey(0)

        # process depth image
        cur_depth = ros2_numpy.numpify(depth)

        # get mask using DloSegmenter
        result = self.segmenter.segment(cur_image)

        try:
            mask_mono = result.mask.copy().astype(np.uint8)
            # Normalize mask to binary 0/255.
            if int(mask_mono.max()) <= 1:
                mask_mono = (mask_mono > 0).astype(np.uint8) * 255
            else:
                _, mask_mono = cv2.threshold(mask_mono, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Cable foreground should be sparse; invert if mask is likely flipped.
            foreground_ratio = float(np.count_nonzero(mask_mono)) / float(mask_mono.size)
            if foreground_ratio > 0.5:
                mask_mono = cv2.bitwise_not(mask_mono)

            mask_msg = self.bridge.cv2_to_imgmsg(mask_mono, encoding='mono8')
            mask_msg.header = rgb.header
            self.mask_pub.publish(mask_msg)

            # TrackDLO only consumes the first init_nodes message.
            # Keep publishing masks, but avoid re-initializing once nodes are latched.
            if self.init_nodes_published:
                return

            #cv2.imshow('Mask', mask)
            #cv2.waitKey(0)

            mask_bgr = cv2.cvtColor(mask_mono, cv2.COLOR_GRAY2BGR)
            extracted_chains = extract_connected_skeleton(
                self.visualize_initialization_process,
                mask_bgr,
                img_scale=1,
                seg_length=8,
                max_curvature=25,
            )

            ordered_centerline_rc = self.merge_extracted_chains(extracted_chains)
            if ordered_centerline_rc.shape[0] < self.num_of_nodes:
                self.get_logger().warn('Chain extraction/merge failed or too short; waiting for a better frame.')
                return

            sampled_pixels_rc, sampled_idx = self.sample_polyline_pixels_by_arclength(ordered_centerline_rc, self.num_of_nodes)
            if sampled_pixels_rc.shape[0] < self.num_of_nodes:
                self.get_logger().warn('Failed to sample enough centerline pixels; waiting for a better frame.')
                return

            init_nodes = self.centerline_pixels_to_3d_nodes(ordered_centerline_rc, sampled_idx, cur_depth)
            if init_nodes.shape[0] < self.num_of_nodes:
                self.get_logger().warn('Depth along centerline is insufficient for full initialization; waiting for a better frame.')
                return

            self.publish_init_debug_image(rgb, cur_image, mask_mono, sampled_pixels_rc)

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
            self.init_nodes_published = True
            self.get_logger().info('Published stable initialization nodes.')
 
        except Exception as e:
            self.get_logger().error(e)
            self.get_logger().error("Failed to generate mask-constrained initialization nodes.")
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    tracker_initializer = TrackerInitializer()
    rclpy.spin(tracker_initializer)

    tracker_initializer.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()
    
