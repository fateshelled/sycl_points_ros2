#pragma once

#include <tf2_ros/transform_broadcaster.h>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sycl_points/algorithms/knn_search.hpp>
#include <sycl_points/algorithms/preprocess_filter.hpp>
#include <sycl_points/algorithms/registration.hpp>
#include <sycl_points/algorithms/voxel_downsampling.hpp>
#include <sycl_points/utils/sycl_utils.hpp>

namespace sycl_points {
namespace ros2 {
class LiDAROdometryNode : public rclcpp::Node {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LiDAROdometryNode(const rclcpp::NodeOptions& options);
    ~LiDAROdometryNode();

    struct Parameters {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        std::string sycl_device_vendor = "intel";
        std::string sycl_device_type = "gpu";
        float scan_voxel_size = 1.0f;
        int32_t scan_covariance_neighbor_num = 10;
        float scan_preprocess_box_filter_min = 2.0f;
        float scan_preprocess_box_filter_max = 50.0f;
        int32_t scan_preprocess_random_sampling_num = 1000;

        float submap_voxel_size = 1.0f;
        int32_t submap_covariance_neighbor_num = 10;
        float keyframe_inlier_ratio_threshold = 0.7f;
        float keyframe_distance_threshold = 2.0f;
        float keyframe_angle_threshold_degrees = 20.0f;

        size_t gicp_min_num_points = 100;
        algorithms::registration::RegistrationParams gicp;

        std::string odom_frame_id = "odom";
        std::string base_link_id = "base_link";

        Eigen::Isometry3f T_base_link_to_lidar = Eigen::Isometry3f::Identity();
        Eigen::Isometry3f initial_pose = Eigen::Isometry3f::Identity();
    };

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pc_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_preprocessed_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_submap_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_pose_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    sycl_utils::DeviceQueue::Ptr queue_ptr_;

    PointCloudShared::Ptr scan_pc_ = nullptr;
    PointCloudShared::Ptr preprocessed_pc_ = nullptr;
    PointCloudShared::Ptr submap_pc_ = nullptr;
    algorithms::knn_search::KDTree::Ptr submap_tree_ = nullptr;

    algorithms::filter::PreprocessFilter::Ptr preprocess_filter_;
    algorithms::filter::VoxelGrid::Ptr voxel_filter_;
    algorithms::filter::VoxelGrid::Ptr submap_voxel_filter_;
    algorithms::registration::RegistrationGICP::Ptr gicp_;

    Eigen::Isometry3f odom_;
    Eigen::Isometry3f last_keyframe_pose_;

    Parameters params_;

    Parameters get_parameters();
    void imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr msg);
    void point_cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);
    void publish_odom(const std_msgs::msg::Header& header, const Eigen::Isometry3f& odom);
};
}  // namespace ros2
}  // namespace sycl_points
