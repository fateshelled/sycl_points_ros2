from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch.actions import TimerAction
import os
import yaml


def declare_params_from_yaml(yaml_path: str, target_node="lidar_odometry_node"):
    launch_args = []
    node_args = {}
    with open(yaml_path, "r") as f:
        all_params = yaml.safe_load(f)

    for node_name in all_params.keys():
        if node_name == target_node:
            node_params: dict = all_params[node_name]["ros__parameters"]
            for name, value in node_params.items():
                launch_args.append(
                    DeclareLaunchArgument(
                        name, default_value=str(value), description=""
                    )
                )
                node_args[name] = LaunchConfiguration(name)
            break
    return launch_args, node_args


def generate_launch_description():
    package_name = "sycl_points_ros2"
    node_name = "lidar_odometry_node"
    package_dir = get_package_share_directory(package_name)
    param_yaml = os.path.join(package_dir, "config", "lidar_odometry.yaml")
    launch_args, node_args = declare_params_from_yaml(param_yaml, node_name)
    launch_args.extend(
        [
            DeclareLaunchArgument(
                "point_topic",
                default_value="/os_cloud_node/points",
                description="source point cloud topic",
            ),
            DeclareLaunchArgument(
                "lidar_frame_id",
                default_value="os_sensor",
                description="source point cloud frame id",
            ),
            DeclareLaunchArgument(
                "rviz2",
                default_value="true",
                choices=["true", "false"],
                description="launch with rviz2",
            ),
            DeclareLaunchArgument(
                "odom_frame_id",
                default_value="odom",
                description="odom frame id",
            ),
            DeclareLaunchArgument(
                "base_link_id",
                default_value="base_link",
                description="base_link frame id",
            ),
            DeclareLaunchArgument(
                "base_link_to_lidar_frame.x",
                default_value="0.0",
                description="static transform x from base_link to lidar_frame",
            ),
            DeclareLaunchArgument(
                "base_link_to_lidar_frame.y",
                default_value="0.0",
                description="static transform y from base_link to lidar_frame",
            ),
            DeclareLaunchArgument(
                "base_link_to_lidar_frame.z",
                default_value="0.0",
                description="static transform z from base_link to lidar_frame",
            ),
            DeclareLaunchArgument(
                "base_link_to_lidar_frame.qx",
                default_value="0.0",
                description="static transform qx from base_link to lidar_frame",
            ),
            DeclareLaunchArgument(
                "base_link_to_lidar_frame.qy",
                default_value="0.0",
                description="static transform qy from base_link to lidar_frame",
            ),
            DeclareLaunchArgument(
                "base_link_to_lidar_frame.qz",
                default_value="0.0",
                description="static transform qz from base_link to lidar_frame",
            ),
            DeclareLaunchArgument(
                "base_link_to_lidar_frame.qw",
                default_value="1.0",
                description="static transform qw from base_link to lidar_frame",
            ),
        ]
    )

    nodes = [
        Node(
            package=package_name,
            executable=node_name,
            name=node_name,
            output="screen",
            emulate_tty=True,
            parameters=[
                node_args,
                {
                    "odom_frame_id": LaunchConfiguration("odom_frame_id"),
                    "base_link_id": LaunchConfiguration("base_link_id"),
                },
            ],
            remappings=[
                ("points", LaunchConfiguration("point_topic")),
            ],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", os.path.join(package_dir, "rviz2", "rviz2.rviz")],
            condition=IfCondition(LaunchConfiguration("rviz2")),
        ),
        TimerAction(
            period=1.0,
            actions=[
                Node(
                    package="tf2_ros",
                    executable="static_transform_publisher",
                    arguments=["--x", LaunchConfiguration("base_link_to_lidar_frame.x")]
                    + ["--y", LaunchConfiguration("base_link_to_lidar_frame.y")]
                    + ["--z", LaunchConfiguration("base_link_to_lidar_frame.z")]
                    + ["--qx", LaunchConfiguration("base_link_to_lidar_frame.qx")]
                    + ["--qy", LaunchConfiguration("base_link_to_lidar_frame.qy")]
                    + ["--qz", LaunchConfiguration("base_link_to_lidar_frame.qz")]
                    + ["--qw", LaunchConfiguration("base_link_to_lidar_frame.qw")]
                    + ["--frame-id", "base_link"]
                    + ["--child-frame-id", LaunchConfiguration("lidar_frame_id")],
                ),
            ],
        ),
    ]

    return LaunchDescription(launch_args + nodes)
