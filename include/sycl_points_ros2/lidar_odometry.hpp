#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sycl_points/algorithms/preprocess_filter.hpp>
#include <sycl_points/algorithms/registration.hpp>
#include <sycl_points/algorithms/voxel_downsampling.hpp>
#include <sycl_points/utils/sycl_utils.hpp>

namespace sycl_points {
namespace ros2 {
class LiDAROdometryNode : public rclcpp::Node {
public:
    LiDAROdometryNode(const rclcpp::NodeOptions &options);
    ~LiDAROdometryNode();

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pc_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_preprocessed_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_submap_;
    sycl_points::sycl_utils::DeviceQueue::Ptr queue_ptr_;

    sycl_points::PointCloudShared::Ptr scan_pc_ = nullptr;
    sycl_points::PointCloudShared::Ptr preprocessed_pc_ = nullptr;

    sycl_points::algorithms::filter::PreprocessFilter::Ptr preprocess_filter_;
    sycl_points::algorithms::filter::VoxelGrid::Ptr voxel_filter_;
    sycl_points::algorithms::registration::RegistrationGICP::Ptr gicp_;
    sycl_points::algorithms::registration::RegistrationParams gicp_param_;

    void point_cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);
};
}  // namespace ros2
}  // namespace sycl_points