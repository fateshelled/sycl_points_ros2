#include "sycl_points_ros2/lidar_odometry.hpp"

#include <rclcpp_components/register_node_macro.hpp>
#include <sycl_points/ros2/convert.hpp>

namespace sycl_points {
namespace ros2 {

/// @brief constructor
/// @param options node option
LiDAROdometryNode::LiDAROdometryNode(const rclcpp::NodeOptions& options) : rclcpp::Node("lidar_odometry", options) {

    const auto device_selector = sycl_points::sycl_utils::device_selector::default_selector_v;
    sycl::device dev(device_selector);
    this->queue_ptr_ = std::make_shared<sycl_points::sycl_utils::DeviceQueue>(dev);
    this->queue_ptr_->print_device_info();

    this->sub_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/camera/depth/color/points", rclcpp::SensorDataQoS(),
        std::bind(&LiDAROdometryNode::point_cloud_callback, this, std::placeholders::_1));

    this->pub_preprocessed_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>("sycl_lo/preprocessed", rclcpp::SensorDataQoS());
    this->pub_submap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("sycl_lo/submap", rclcpp::SensorDataQoS());
}

/// @brief destructor
LiDAROdometryNode::~LiDAROdometryNode() {}

void LiDAROdometryNode::point_cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
    const auto scan = fromROS2msg(*this->queue_ptr_, *msg);

    auto pub_msg = toROS2msg(*scan, msg->header);
    this->pub_preprocessed_->publish(std::move(pub_msg));
}

}  // namespace ros2
}  // namespace sycl_points

RCLCPP_COMPONENTS_REGISTER_NODE(sycl_points::ros2::LiDAROdometryNode)