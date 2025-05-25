#include "sycl_points_ros2/lidar_odometry.hpp"

#include <rclcpp_components/register_node_macro.hpp>
#include <sycl_points/ros2/convert.hpp>
#include <sycl_points/utils/time_utils.hpp>

namespace sycl_points {
namespace ros2 {

/// @brief constructor
/// @param options node option
LiDAROdometryNode::LiDAROdometryNode(const rclcpp::NodeOptions& options) : rclcpp::Node("lidar_odometry", options) {
    // SYCL queue
    const auto device_selector = sycl_utils::device_selector::default_selector_v;
    sycl::device dev(device_selector);
    this->queue_ptr_ = std::make_shared<sycl_utils::DeviceQueue>(dev);
    this->queue_ptr_->print_device_info();

    // initialize params
    const float voxel_size = 1.00f;
    this->gicp_param_.max_iterations = 10;
    this->gicp_param_.max_correspondence_distance = 1.0f;
    this->gicp_param_.verbose = false;

    // Point cloud processor
    this->preprocess_filter_ = std::make_shared<algorithms::filter::PreprocessFilter>(*this->queue_ptr_);
    this->voxel_filter_ = std::make_shared<algorithms::filter::VoxelGrid>(*this->queue_ptr_, voxel_size);
    this->gicp_ = std::make_shared<algorithms::registration::RegistrationGICP>(*this->queue_ptr_, this->gicp_param_);

    // pub/sub
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
    time_utils::measure_execution([&]() { return fromROS2msg(*this->queue_ptr_, *msg, this->scan_pc_); },
                                  dt_from_ros2_msg);
    double dt_voxel_downsampling = 0.0;
    time_utils::measure_execution(
        [&]() {
            if (this->preprocessed_pc_ == nullptr) {
                this->preprocessed_pc_ = std::make_shared<PointCloudShared>(*this->queue_ptr_);
            }
            this->voxel_filter_->downsampling(*this->scan_pc_, *this->preprocessed_pc_);
        },
        dt_voxel_downsampling);

    double dt_to_ros2_msg = 0.0;
    auto pub_msg = time_utils::measure_execution([&]() { return toROS2msg(*this->preprocessed_pc_, msg->header); },
                                                 dt_to_ros2_msg);

    double dt_publish = 0.0;
    if (this->pub_preprocessed_->get_subscription_count() > 0) {
        time_utils::measure_execution([&]() { return this->pub_preprocessed_->publish(std::move(pub_msg)); },
                                      dt_publish);
    }

    RCLCPP_INFO(this->get_logger(), "fromROS2msg:       %8.3f us", dt_from_ros2_msg);
    RCLCPP_INFO(this->get_logger(), "VoxelDownsampling: %8.3f us", dt_voxel_downsampling);
    RCLCPP_INFO(this->get_logger(), "toROS2msg:         %8.3f us", dt_to_ros2_msg);
    RCLCPP_INFO(this->get_logger(), "publish:           %8.3f us", dt_publish);
}

}  // namespace ros2
}  // namespace sycl_points

RCLCPP_COMPONENTS_REGISTER_NODE(sycl_points::ros2::LiDAROdometryNode)