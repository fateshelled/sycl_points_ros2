from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch.actions import TimerAction
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
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
                if isinstance(value, float):
                    value_str = format(value, "f")
                else:
                    value_str = str(value)
                launch_args.append(
                    DeclareLaunchArgument(name, default_value=value_str, description="")
                )
                node_args[name] = LaunchConfiguration(name)
            break
    return launch_args, node_args


def get_T_base_link_to_lidar(yaml_path: str, target_node="lidar_odometry_node"):
    with open(yaml_path, "r") as f:
        all_params = yaml.safe_load(f)

    for node_name in all_params.keys():
        if node_name == target_node:
            node_params: dict = all_params[node_name]["ros__parameters"]
            for name, value in node_params.items():
                if name == "T_base_link_to_lidar":
                    return value
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def generate_launch_description():
    package_name = "sycl_points_ros2"
    node_name = "lidar_odometry_node"
    package_dir = get_package_share_directory(package_name)
    param_yaml = os.path.join(package_dir, "config", "lidar_odometry.yaml")
    launch_args, node_args = declare_params_from_yaml(param_yaml, node_name)
    T_base_link_to_lidar = get_T_base_link_to_lidar(param_yaml, node_name)
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
        ]
    )

    nodes = [
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
                    arguments=["--x", str(T_base_link_to_lidar[0])]
                    + ["--y", str(T_base_link_to_lidar[1])]
                    + ["--z", str(T_base_link_to_lidar[2])]
                    + ["--qx", str(T_base_link_to_lidar[3])]
                    + ["--qy", str(T_base_link_to_lidar[4])]
                    + ["--qz", str(T_base_link_to_lidar[5])]
                    + ["--qw", str(T_base_link_to_lidar[6])]
                    + ["--frame-id", "base_link"]
                    + ["--child-frame-id", LaunchConfiguration("lidar_frame_id")],
                ),
                ComposableNodeContainer(
                    name="sycl_points_container",
                    namespace="",
                    package="rclcpp_components",
                    executable="component_container",    # SingleThreadedExecutor
                    # executable="component_container_mt",  # MultiThreadedExecutor
                    output="screen",
                    emulate_tty=True,
                    composable_node_descriptions=[
                        ComposableNode(
                            package=package_name,
                            plugin="sycl_points::ros2::LiDAROdometryNode",
                            name=package_name,
                            parameters=[
                                node_args,
                                {
                                    "odom_frame_id": LaunchConfiguration(
                                        "odom_frame_id"
                                    ),
                                    "base_link_id": LaunchConfiguration("base_link_id"),
                                },
                            ],
                            remappings=[
                                ("points", LaunchConfiguration("point_topic")),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ]

    return LaunchDescription(launch_args + nodes)
