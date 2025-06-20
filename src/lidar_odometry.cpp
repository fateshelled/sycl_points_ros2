#include "sycl_points_ros2/lidar_odometry.hpp"

#include <rclcpp_components/register_node_macro.hpp>
#include <sycl_points/algorithms/covariance.hpp>
#include <sycl_points/ros2/convert.hpp>
#include <sycl_points/utils/time_utils.hpp>

namespace sycl_points {
namespace ros2 {

/// @brief constructor
/// @param options node option
LiDAROdometryNode::LiDAROdometryNode(const rclcpp::NodeOptions& options) : rclcpp::Node("lidar_odometry", options) {
    this->params_ = this->get_parameters();

    // SYCL queue
    // const auto device_selector = sycl_utils::device_selector::default_selector_v;
    // sycl::device dev(device_selector);
    const auto dev =
        sycl_utils::device_selector::select_device(this->params_.sycl_device_vendor, this->params_.sycl_device_type);
    this->queue_ptr_ = std::make_shared<sycl_utils::DeviceQueue>(dev);
    this->queue_ptr_->print_device_info();

    // set Initial pose
    this->odom_ = this->params_.initial_pose;
    this->last_keyframe_pose_ = this->params_.initial_pose;

    // Point cloud processor
    this->preprocess_filter_ = std::make_shared<algorithms::filter::PreprocessFilter>(*this->queue_ptr_);
    this->voxel_filter_ =
        std::make_shared<algorithms::filter::VoxelGrid>(*this->queue_ptr_, this->params_.scan_voxel_size);
    this->submap_voxel_filter_ =
        std::make_shared<algorithms::filter::VoxelGrid>(*this->queue_ptr_, this->params_.submap_voxel_size);
    this->gicp_ = std::make_shared<algorithms::registration::RegistrationGICP>(*this->queue_ptr_, this->params_.gicp);

    // pub/sub
    this->sub_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "points", rclcpp::QoS(10), std::bind(&LiDAROdometryNode::point_cloud_callback, this, std::placeholders::_1));

    this->pub_preprocessed_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>("sycl_lo/preprocessed", rclcpp::SensorDataQoS());
    this->pub_submap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("sycl_lo/submap", rclcpp::QoS(1));

    this->pub_odom_ = this->create_publisher<nav_msgs::msg::Odometry>("sycl_lo/odom", rclcpp::QoS(10));
    this->pub_pose_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("sycl_lo/pose", rclcpp::QoS(10));
    this->tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    RCLCPP_INFO(this->get_logger(), "Subscribe PointCloud: %s", this->sub_pc_->get_topic_name());
    RCLCPP_INFO(this->get_logger(), "Publish Preprocessed PointCloud: %s", this->pub_preprocessed_->get_topic_name());
    RCLCPP_INFO(this->get_logger(), "Publish Submap PointCloud: %s", this->pub_submap_->get_topic_name());
    RCLCPP_INFO(this->get_logger(), "Publish Odometry: %s", this->pub_odom_->get_topic_name());
    RCLCPP_INFO(this->get_logger(), "Publish Pose: %s", this->pub_pose_->get_topic_name());
}

/// @brief destructor
LiDAROdometryNode::~LiDAROdometryNode() {}

LiDAROdometryNode::Parameters LiDAROdometryNode::get_parameters() {
    LiDAROdometryNode::Parameters params;

    params.sycl_device_vendor = this->declare_parameter<std::string>("sycl/device_vendor", params.sycl_device_vendor);
    params.sycl_device_type = this->declare_parameter<std::string>("sycl/device_type", params.sycl_device_type);

    params.scan_voxel_size = this->declare_parameter<double>("scan/voxel_size", params.scan_voxel_size);
    params.scan_covariance_neighbor_num =
        this->declare_parameter<int>("scan/covariance/neighbor_num", params.scan_covariance_neighbor_num);
    params.scan_preprocess_box_filter_min =
        this->declare_parameter<double>("scan/preprocess/box_filter/min", params.scan_preprocess_box_filter_min);
    params.scan_preprocess_box_filter_max =
        this->declare_parameter<double>("scan/preprocess/box_filter/max", params.scan_preprocess_box_filter_max);
    params.scan_preprocess_random_sampling_num =
        this->declare_parameter<int>("scan/preprocess/random_sampling/num", params.scan_preprocess_random_sampling_num);

    params.submap_voxel_size = this->declare_parameter<double>("submap/voxel_size", params.submap_voxel_size);
    params.submap_covariance_neighbor_num =
        this->declare_parameter<int>("submap/covariance/neighbor_num", params.submap_covariance_neighbor_num);

    params.keyframe_inlier_ratio_threshold =
        this->declare_parameter<double>("keyframe/inlier_ratio_threshold", params.keyframe_inlier_ratio_threshold);
    params.keyframe_distance_threshold =
        this->declare_parameter<double>("keyframe/distance_threshold", params.keyframe_distance_threshold);
    params.keyframe_angle_threshold_degrees =
        this->declare_parameter<double>("keyframe/angle_threshold_degrees", params.keyframe_angle_threshold_degrees);

    params.gicp.max_iterations = this->declare_parameter<double>("gicp/max_iterations", 20);
    params.gicp.lambda = this->declare_parameter<double>("gicp/lambda", 1e-4);
    params.gicp.max_correspondence_distance = this->declare_parameter<double>("gicp/max_correspondence_distance", 2.0);
    params.gicp.adaptive_correspondence_distance =
        this->declare_parameter<bool>("gicp/adaptive_correspondence_distance", true);
    params.gicp.inlier_ratio = this->declare_parameter<double>("gicp/inlier_ratio", 0.7);
    params.gicp.translation_eps = this->declare_parameter<double>("gicp/translation_eps", 1e-3);
    params.gicp.rotation_eps = this->declare_parameter<double>("gicp/rotation_eps", 1e-3);

    const std::string robust_loss = this->declare_parameter<std::string>("gicp/robust_loss", "NONE");
    params.gicp.robust_loss = algorithms::registration::RobustLossType_from_string(robust_loss);
    params.gicp.robust_threshold = this->declare_parameter<double>("gicp/robust_threshold", 1.0);

    params.gicp.verbose = this->declare_parameter<bool>("gicp/verbose", true);

    {
        // Ouster os0
        // x, y, z, qx, qy, qz, qw
        const auto T =
            this->declare_parameter<std::vector<double>>("T_base_link_to_lidar", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0});
        if (T.size() != 7) throw std::runtime_error("invalid T_base_link_to_lidar");
        params.T_base_link_to_lidar.setIdentity();
        params.T_base_link_to_lidar.translation() << T[0], T[1], T[2];
        const Eigen::Quaternionf quat(T[6], T[3], T[4], T[5]);
        params.T_base_link_to_lidar.matrix().block<3, 3>(0, 0) = quat.matrix();
    }

    {
        // x, y, z, qx, qy, qz, qw
        const auto T =
            this->declare_parameter<std::vector<double>>("initial_pose", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0});
        if (T.size() != 7) throw std::runtime_error("invalid initial_pose");
        params.initial_pose.setIdentity();
        params.initial_pose.translation() << T[0], T[1], T[2];
        const Eigen::Quaternionf quat(T[6], T[3], T[4], T[5]);
        params.initial_pose.matrix().block<3, 3>(0, 0) = quat.matrix();
    }
    return params;
}

void LiDAROdometryNode::point_cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
    const auto timestamp = rclcpp::Time(msg->header.stamp).seconds();

    double dt_from_ros2_msg = 0.0;
    time_utils::measure_execution([&]() { return fromROS2msg(*this->queue_ptr_, *msg, this->scan_pc_); },
                                  dt_from_ros2_msg);

    // preprocess
    double dt_preprocessing = 0.0;
    time_utils::measure_execution(
        [&]() {
            if (this->preprocessed_pc_ == nullptr) {
                this->preprocessed_pc_ = std::make_shared<PointCloudShared>(*this->queue_ptr_);
            }
            this->preprocess_filter_->box_filter(*this->scan_pc_, this->params_.scan_preprocess_box_filter_min,
                                                 this->params_.scan_preprocess_box_filter_max);
            this->voxel_filter_->downsampling(*this->scan_pc_, *this->preprocessed_pc_);
        },
        dt_preprocessing);

    // compute covariances
    double dt_covariance = 0.0;
    const auto src_tree = time_utils::measure_execution(
        [&]() {
            const auto tree =
                sycl_points::algorithms::knn_search::KDTree::build(*this->queue_ptr_, *this->preprocessed_pc_);
            algorithms::covariance::compute_covariances_async(*tree, *this->preprocessed_pc_,
                                                              this->params_.scan_covariance_neighbor_num)
                .wait();
            algorithms::covariance::covariance_update_plane(*this->preprocessed_pc_);
            return tree;
        },
        dt_covariance);

    // is first frame
    if (this->submap_pc_ == nullptr) {
        // copy to submap
        this->submap_pc_ = std::make_shared<PointCloudShared>(*this->preprocessed_pc_);
        this->submap_tree_ = src_tree;
        return;
    }

    // Registration
    double dt_registration = 0.0;
    const auto reg_result = time_utils::measure_execution(
        [&]() {
            const Eigen::Isometry3f init_T = this->odom_;

            this->preprocess_filter_->random_sampling(*this->preprocessed_pc_,
                                                      this->params_.scan_preprocess_random_sampling_num);

            const auto result =
                this->gicp_->align(*this->preprocessed_pc_, *this->submap_pc_, *this->submap_tree_, init_T.matrix());

            // update odometry
            this->odom_ = result.T;
            return result;
        },
        dt_registration);

    // Build submap
    double dt_build_submap = 0.0;
    bool update_submap = false;
    time_utils::measure_execution(
        [&]() {
            const float inlier_ratio = static_cast<float>(reg_result.inlier) / this->preprocessed_pc_->size();
            if (inlier_ratio <= this->params_.keyframe_inlier_ratio_threshold) {
                return;
            }

            // calculate delta pose
            const auto delta_pose = this->last_keyframe_pose_.inverse() * reg_result.T;

            // calculate moving distance and angle
            const auto distance = delta_pose.translation().norm();
            const auto angle = Eigen::AngleAxisf(delta_pose.rotation()).angle() * (float)(180.0 / M_PI);

            // update submap
            if (distance >= this->params_.keyframe_distance_threshold ||
                angle >= this->params_.keyframe_angle_threshold_degrees) {
                this->last_keyframe_pose_ = reg_result.T;

                const auto aligned = this->gicp_->get_aligned_point_cloud();
                if (aligned == nullptr) {
                    RCLCPP_ERROR(this->get_logger(), "aligned pc is nullptr");
                    return;
                }
                // add new points
                this->submap_pc_->extend(*aligned);

                // Voxel downsampling
                this->submap_voxel_filter_->downsampling(*this->submap_pc_, *this->submap_pc_);

                this->submap_tree_ =
                    sycl_points::algorithms::knn_search::KDTree::build(*this->queue_ptr_, *this->submap_pc_);

                algorithms::covariance::compute_covariances_async(*this->submap_tree_, *this->submap_pc_,
                                                                  this->params_.submap_covariance_neighbor_num)
                    .wait();
                algorithms::covariance::covariance_update_plane(*this->submap_pc_);
                update_submap = true;
            }
        },
        dt_build_submap);

    // publish ROS2 message
    double dt_to_ros2_msg = 0.0;
    auto preprocessed_msg = time_utils::measure_execution(
        [&]() { return std::move(toROS2msg(*this->preprocessed_pc_, msg->header)); }, dt_to_ros2_msg);
    auto submap_msg = time_utils::measure_execution(
        [&]() {
            if (update_submap) {
                auto submap_msg = toROS2msg(*this->submap_pc_, msg->header);
                submap_msg->header.frame_id = "odom";
                return std::move(submap_msg);
            }
            sensor_msgs::msg::PointCloud2::UniquePtr ret = nullptr;
            return std::move(ret);
        },
        dt_to_ros2_msg);

    double dt_publish = 0.0;
    time_utils::measure_execution(
        [&]() {
            this->publish_odom(msg->header, this->odom_);
            if (preprocessed_msg != nullptr && this->pub_preprocessed_->get_subscription_count() > 0) {
                this->pub_preprocessed_->publish(std::move(preprocessed_msg));
            }
            if (submap_msg != nullptr && this->pub_submap_->get_subscription_count() > 0) {
                this->pub_submap_->publish(std::move(submap_msg));
            }
        },
        dt_publish);

    RCLCPP_INFO(this->get_logger(), "fromROS2msg:         %9.2f us", dt_from_ros2_msg);
    RCLCPP_INFO(this->get_logger(), "Preprocessing:       %9.2f us", dt_preprocessing);
    RCLCPP_INFO(this->get_logger(), "compute Covariances: %9.2f us", dt_covariance);
    RCLCPP_INFO(this->get_logger(), "Registration:        %9.2f us", dt_registration);
    RCLCPP_INFO(this->get_logger(), "Build submap:        %9.2f us", dt_build_submap);
    RCLCPP_INFO(this->get_logger(), "toROS2msg:           %9.2f us", dt_to_ros2_msg);
    RCLCPP_INFO(this->get_logger(), "publish:             %9.2f us", dt_publish);
}

void LiDAROdometryNode::publish_odom(const std_msgs::msg::Header& header, const Eigen::Isometry3f& odom) {
    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = header.stamp;
    tf.header.frame_id = "odom";
    tf.child_frame_id = "base_link";

    const auto odom_trans = odom.translation();
    tf.transform.translation.x = odom_trans.x();
    tf.transform.translation.y = odom_trans.y();
    tf.transform.translation.z = odom_trans.z();
    const Eigen::Quaternionf odom_quat(odom.rotation());
    tf.transform.rotation.x = odom_quat.x();
    tf.transform.rotation.y = odom_quat.y();
    tf.transform.rotation.z = odom_quat.z();
    tf.transform.rotation.w = odom_quat.w();
    this->tf_broadcaster_->sendTransform(tf);

    geometry_msgs::msg::PoseStamped pose;
    pose.header = tf.header;
    pose.pose.position.x = odom_trans.x();
    pose.pose.position.y = odom_trans.y();
    pose.pose.position.z = odom_trans.z();
    pose.pose.orientation.x = odom_quat.x();
    pose.pose.orientation.y = odom_quat.y();
    pose.pose.orientation.z = odom_quat.z();
    pose.pose.orientation.w = odom_quat.w();
    this->pub_pose_->publish(pose);

    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header = tf.header;
    odom_msg.child_frame_id = tf.child_frame_id;
    odom_msg.pose.pose.position = pose.pose.position;
    odom_msg.pose.pose.orientation = pose.pose.orientation;
    this->pub_odom_->publish(odom_msg);
}

}  // namespace ros2
}  // namespace sycl_points

RCLCPP_COMPONENTS_REGISTER_NODE(sycl_points::ros2::LiDAROdometryNode)