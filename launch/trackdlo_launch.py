#!/usr/bin/env python3

import os

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter


def generate_launch_description():

    args = [
        DeclareLaunchArgument("name", default_value="trackdlo", description="node name"),
        DeclareLaunchArgument("namespace", default_value="", description="node namespace"),
        DeclareLaunchArgument("params", default_value=os.path.join(get_package_share_directory("trackdlo"), "config", "trackdlo_params.yml"), description="path to parameter file"),
        DeclareLaunchArgument("log_level", default_value="info", description="ROS logging level (debug, info, warn, error, fatal)"),
        DeclareLaunchArgument("use_sim_time", default_value="false", description="use simulation clock")
    ]

    nodes = [
        Node(
            package="trackdlo",
            executable="trackdlo",
            namespace=LaunchConfiguration("namespace"),
            name=LaunchConfiguration("name"),
            parameters=[LaunchConfiguration("params")],
            arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
            output="screen",
            emulate_tty=True
        ),
        Node(
            package="trackdlo",
            name="init_tracker",
            executable="initialize.py",
            parameters=[LaunchConfiguration("params")]
        )
    ]

    return LaunchDescription([
        *args,
        SetParameter("use_sim_time", LaunchConfiguration("use_sim_time")),
        *nodes,
    ])
