#pragma once

#include <tf2_ros/transform_broadcaster.h>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sycl_points/algorithms/mapping/occupancy_grid_map.hpp>
#include <sycl_points/algorithms/mapping/voxel_hash_map.hpp>
#include <sycl_points/algorithms/knn/kdtree.hpp>
#include <sycl_points/algorithms/polar_downsampling.hpp>
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
        bool scan_downsampling_voxel_enable = true;
        float scan_downsampling_voxel_size = 1.0f;
        bool scan_downsampling_polar_enable = false;
        float scan_downsampling_polar_distance_size = 1.0f;
        float scan_downsampling_polar_elevation_size = 3.0f * M_PIf / 180.0f;
        float scan_downsampling_polar_azimuth_size = 3.0f * M_PIf / 180.0f;
        std::string scan_downsampling_polar_coord_system = "CAMERA";

        int32_t scan_covariance_neighbor_num = 10;
        bool scan_preprocess_box_filter_enable = true;
        float scan_preprocess_box_filter_min = 2.0f;
        float scan_preprocess_box_filter_max = 50.0f;
        bool scan_preprocess_random_sampling_enable = true;
        int32_t scan_preprocess_random_sampling_num = 1000;

        float submap_voxel_size = 1.0f;
        int32_t submap_covariance_neighbor_num = 10;
        int32_t submap_color_gradient_neighbor_num = 10;
        float keyframe_inlier_ratio_threshold = 0.7f;
        float keyframe_distance_threshold = 2.0f;
        float keyframe_angle_threshold_degrees = 20.0f;
        float keyframe_time_threshold_seconds = 1.0f;

        bool occupancy_grid_map_enable = true;
        float occupancy_grid_map_log_odds_hit = 0.8;
        float occupancy_grid_map_log_odds_miss = -0.05;
        float occupancy_grid_map_log_odds_limits_min = -1.0;
        float occupancy_grid_map_log_odds_limits_max = 4.0;
        float occupancy_grid_map_occupied_threshold = 0.5;
        float occupancy_grid_map_visibility_decay_range = 100.0;

        size_t gicp_min_num_points = 100;
        algorithms::registration::RegistrationParams gicp;

        std::string odom_frame_id = "odom";
        std::string base_link_id = "base_link";

        Eigen::Isometry3f T_base_link_to_lidar = Eigen::Isometry3f::Identity();
        Eigen::Isometry3f T_lidar_to_base_link = Eigen::Isometry3f::Identity();

        Eigen::Isometry3f initial_pose = Eigen::Isometry3f::Identity();  // map to base_link
    };

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pc_ = nullptr;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_preprocessed_ = nullptr;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_submap_ = nullptr;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_pose_ = nullptr;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_keyframe_pose_ = nullptr;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_ = nullptr;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_ = nullptr;

    sycl_utils::DeviceQueue::Ptr queue_ptr_ = nullptr;

    sycl_points::shared_vector_ptr<uint8_t> msg_data_buffer_ = nullptr;
    PointCloudShared::Ptr scan_pc_ = nullptr;
    PointCloudShared::Ptr preprocessed_pc_ = nullptr;
    PointCloudShared::Ptr gicp_input_pc_ = nullptr;
    PointCloudShared::Ptr submap_pc_ = nullptr;

    algorithms::mapping::VoxelHashMap::Ptr submap_voxel_ = nullptr;
    algorithms::mapping::OccupancyGridMap::Ptr occupancy_grid_ = nullptr;
    algorithms::knn::KDTree::Ptr submap_tree_ = nullptr;
    algorithms::knn::KNNResult knn_result_;

    algorithms::filter::PreprocessFilter::Ptr preprocess_filter_ = nullptr;
    algorithms::filter::VoxelGrid::Ptr voxel_filter_ = nullptr;
    algorithms::filter::PolarGrid::Ptr polar_filter_ = nullptr;
    algorithms::registration::RegistrationGICP::Ptr gicp_ = nullptr;

    Eigen::Isometry3f odom_;
    Eigen::Isometry3f last_keyframe_pose_;
    double last_keyframe_time_;

    Parameters params_;

    std::map<std::string, std::vector<double>> processing_times_;
    void add_delta_time(const std::string& name, double dt) {
        if (this->processing_times_.count(name) > 0) {
            this->processing_times_[name].push_back(dt);
        } else {
            this->processing_times_[name] = {dt};
        }
    }

    Parameters get_parameters();
    void point_cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);
    void publish_odom(const std_msgs::msg::Header& header, const Eigen::Isometry3f& odom,
                      const algorithms::registration::RegistrationResult& reg_result);
    void publish_keyframe_pose(const std_msgs::msg::Header& header, const Eigen::Isometry3f& odom);
};
}  // namespace ros2
}  // namespace sycl_points
