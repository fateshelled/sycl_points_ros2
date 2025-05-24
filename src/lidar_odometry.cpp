#include "sycl_points_ros2/lidar_odometry.hpp"

#include <rclcpp_components/register_node_macro.hpp>
#include <sycl_points/ros2/convert.hpp>
#include <sycl_points/utils/time_utils.hpp>

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
        "points", rclcpp::SensorDataQoS(),
        std::bind(&LiDAROdometryNode::point_cloud_callback, this, std::placeholders::_1));

    this->pub_preprocessed_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>("sycl_lo/preprocessed", rclcpp::SensorDataQoS());
    this->pub_submap_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>("sycl_lo/submap", rclcpp::SensorDataQoS());
}

/// @brief destructor
LiDAROdometryNode::~LiDAROdometryNode() {}

void LiDAROdometryNode::point_cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
    double dt_from_ros2_msg = 0.0;
    const auto scan = sycl_points::time_utils::measure_execution([&]() { return fromROS2msg(*this->queue_ptr_, *msg); },
                                                                 dt_from_ros2_msg);

    double dt_to_ros2_msg = 0.0;
    auto pub_msg =
        sycl_points::time_utils::measure_execution([&]() { return toROS2msg(*scan, msg->header); }, dt_to_ros2_msg);

    double dt_publish = 0.0;
    if (this->pub_preprocessed_->get_subscription_count() > 0) {
        sycl_points::time_utils::measure_execution(
            [&]() { return this->pub_preprocessed_->publish(std::move(pub_msg)); }, dt_publish);
    }

    RCLCPP_INFO(this->get_logger(), "fromROS2msg: %8.3f us", dt_from_ros2_msg);
    RCLCPP_INFO(this->get_logger(), "toROS2msg:   %8.3f us", dt_to_ros2_msg);
    RCLCPP_INFO(this->get_logger(), "publish:     %8.3f us", dt_publish);
}

}  // namespace ros2
}  // namespace sycl_points

RCLCPP_COMPONENTS_REGISTER_NODE(sycl_points::ros2::LiDAROdometryNode)