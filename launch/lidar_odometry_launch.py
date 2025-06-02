from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
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
    launch_args.append(
        DeclareLaunchArgument(
            "point_topic", default_value="/os_cloud_node/points", description="source point cloud topic"
        )
    )

    nodes = [
        Node(
            package=package_name,
            executable=node_name,
            name=node_name,
            output="screen",
            emulate_tty=True,
            parameters=[node_args],
            remappings=[
                ("points", LaunchConfiguration("point_topic")),
            ]
        ),
    ]

    return LaunchDescription(launch_args + nodes)
