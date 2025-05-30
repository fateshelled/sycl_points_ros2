#include "sycl_points_ros2/lidar_odometry.hpp"

#include <rclcpp_components/register_node_macro.hpp>
#include <sycl_points/algorithms/covariance.hpp>
#include <sycl_points/ros2/convert.hpp>
#include <sycl_points/utils/time_utils.hpp>

#include "sycl_points_ros2/imu.hpp"

namespace sycl_points {
namespace ros2 {

/// @brief constructor
/// @param options node option
LiDAROdometryNode::LiDAROdometryNode(const rclcpp::NodeOptions& options) : rclcpp::Node("lidar_odometry", options) {
    this->declare_parameters();

    // SYCL queue
    const auto device_selector = sycl_utils::device_selector::default_selector_v;
    sycl::device dev(device_selector);
    this->queue_ptr_ = std::make_shared<sycl_utils::DeviceQueue>(dev);
    this->queue_ptr_->print_device_info();

    // initialize params
    const float voxel_size = 1.00f;
    const float submap_voxel_size = 1.00f;
    this->gicp_param_.lambda = 1e-4f;
    this->gicp_param_.max_iterations = 20;
    this->gicp_param_.max_correspondence_distance = 2.0f;
    this->gicp_param_.verbose = true;

    // set Initial pose
    this->odom_.setIdentity();
    this->last_keyframe_pose_.setIdentity();

    this->T_base_link_to_lidar_.setIdentity();
    this->T_base_link_to_imu_.setIdentity();
    this->T_lidar_to_imu_.setIdentity();
    {
        {
            const auto T = this->get_parameter("T_base_link_to_lidar").as_double_array();
            if (T.size() != 7) {
                throw std::runtime_error("invalid T_base_link_to_lidar");
            }
            this->T_base_link_to_lidar_.translation() << T[0], T[1], T[2];
            const Eigen::Quaternionf quat(T[6], T[3], T[4], T[5]);
            this->T_base_link_to_lidar_.matrix().block<3, 3>(0, 0) = quat.matrix();
        }

        {
            const auto T = this->get_parameter("T_base_link_to_imu").as_double_array();
            if (T.size() != 7) {
                throw std::runtime_error("invalid T_base_link_to_imu");
            }
            this->T_base_link_to_imu_.translation() << T[0], T[1], T[2];
            const Eigen::Quaternionf quat(T[6], T[3], T[4], T[5]);
            this->T_base_link_to_imu_.matrix().block<3, 3>(0, 0) = quat.matrix();
        }
        this->T_lidar_to_imu_ = this->T_base_link_to_lidar_.inverse() * this->T_base_link_to_imu_;
        std::cout << "T_lidar_to_imu\n" << this->T_lidar_to_imu_.matrix() << std::endl;
    }

    // Point cloud processor
    this->preprocess_filter_ = std::make_shared<algorithms::filter::PreprocessFilter>(*this->queue_ptr_);
    this->voxel_filter_ = std::make_shared<algorithms::filter::VoxelGrid>(*this->queue_ptr_, voxel_size);
    this->submap_voxel_filter_ = std::make_shared<algorithms::filter::VoxelGrid>(*this->queue_ptr_, submap_voxel_size);
    this->gicp_ = std::make_shared<algorithms::registration::RegistrationGICP>(*this->queue_ptr_, this->gicp_param_);

    // pub/sub
    this->sub_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "points", rclcpp::QoS(10), std::bind(&LiDAROdometryNode::point_cloud_callback, this, std::placeholders::_1));

    this->pub_preprocessed_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>("sycl_lo/preprocessed", rclcpp::SensorDataQoS());
    this->pub_submap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("sycl_lo/submap", rclcpp::QoS(1));

    this->pub_odom_ = this->create_publisher<nav_msgs::msg::Odometry>("sycl_lo/odom", rclcpp::QoS(10));
    this->pub_pose_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("sycl_lo/pose", rclcpp::QoS(10));
    this->tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
}

/// @brief destructor
LiDAROdometryNode::~LiDAROdometryNode() {}

void LiDAROdometryNode::declare_parameters() {
    // Ouster os0
    // x, y, z, qx, qy, qz, qw
    this->declare_parameter<std::vector<double>>("T_base_link_to_lidar", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0});
    this->declare_parameter<std::vector<double>>("T_base_link_to_imu", {0.006, -0.012, 0.008, 0.0, 0.0, 0.0, 1.0});
}

void LiDAROdometryNode::point_cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
    const auto timestamp = rclcpp::Time(msg->header.stamp).seconds();

    double dt_from_ros2_msg = 0.0;
    time_utils::measure_execution([&]() { return fromROS2msg(*this->queue_ptr_, *msg, this->scan_pc_); },
                                  dt_from_ros2_msg);

    // preprocess
    const float box_min = 2.0f;
    const float box_max = 50.0f;
    double dt_preprocessing = 0.0;
    time_utils::measure_execution(
        [&]() {
            if (this->preprocessed_pc_ == nullptr) {
                this->preprocessed_pc_ = std::make_shared<PointCloudShared>(*this->queue_ptr_);
            }
            this->preprocess_filter_->box_filter(*this->scan_pc_, box_min, box_max);
            this->voxel_filter_->downsampling(*this->scan_pc_, *this->preprocessed_pc_);
        },
        dt_preprocessing);

    // compute covariances
    const size_t covariance_neighbor_num = 10;
    double dt_covariance = 0.0;
    const auto src_tree = time_utils::measure_execution(
        [&]() {
            const auto tree =
                sycl_points::algorithms::knn_search::KDTree::build(*this->queue_ptr_, *this->preprocessed_pc_);
            algorithms::covariance::compute_covariances(*tree, *this->preprocessed_pc_, covariance_neighbor_num);
            algorithms::covariance::covariance_update_plane(*this->preprocessed_pc_);
            return tree;
        },
        dt_covariance);

    if (this->submap_pc_ == nullptr) {
        // copy to submap
        this->submap_pc_ = std::make_shared<PointCloudShared>(*this->preprocessed_pc_);
        this->submap_tree_ = src_tree;
        return;
    }

    // Registration
    const size_t random_sampling_size = 1000;
    double dt_registration = 0.0;
    const auto reg_result = time_utils::measure_execution(
        [&]() {
            const Eigen::Isometry3f init_T = this->odom_;

            this->preprocess_filter_->random_sampling(*this->preprocessed_pc_, random_sampling_size);

            const auto result =
                this->gicp_->align(*this->preprocessed_pc_, *this->submap_pc_, *this->submap_tree_, init_T.matrix());

            // update odometry
            this->odom_ = result.T;
            return result;
        },
        dt_registration);

    // Build submap
    const float inlier_ratio_th = 0.7f;
    double dt_build_submap = 0.0;
    time_utils::measure_execution(
        [&]() {
            const float inlier_ratio = static_cast<float>(reg_result.inlier) / this->preprocessed_pc_->size();
            if (inlier_ratio <= inlier_ratio_th) {
                return;
            }

            // calculate delta pose
            const auto delta_pose = this->last_keyframe_pose_.inverse() * reg_result.T;

            // calculate moving distance and angle
            const auto distance = delta_pose.translation().norm();
            const auto angle = Eigen::AngleAxisf(delta_pose.rotation()).angle() * (float)(180.0 / M_PI);

            const float keyframe_distance_threshold = 2.0f;
            const float keyframe_angle_threshold_degrees = 20.0f;
            const size_t submap_covariance_neighbor_num = 20;

            // update submap
            if (distance >= keyframe_distance_threshold || angle >= keyframe_angle_threshold_degrees) {
                this->last_keyframe_pose_ = reg_result.T;

                this->queue_ptr_->ptr
                    ->submit([&](sycl::handler& h) {
                        h.host_task([&]() {
                            const auto aligned = this->gicp_->get_aligned_point_cloud();
                            if (aligned == nullptr) {
                                RCLCPP_ERROR(this->get_logger(), "aligned pc is nullptr");
                            }
                            // add new points
                            this->submap_pc_->extend(*aligned);

                            // Voxel downsampling
                            this->submap_voxel_filter_->downsampling(*this->submap_pc_, *this->submap_pc_);

                            this->submap_tree_ = sycl_points::algorithms::knn_search::KDTree::build(*this->queue_ptr_,
                                                                                                    *this->submap_pc_);

                            algorithms::covariance::compute_covariances(*this->submap_tree_, *this->submap_pc_,
                                                                        submap_covariance_neighbor_num);
                            algorithms::covariance::covariance_update_plane(*this->submap_pc_);
                        });
                    })
                    .wait();
            }
        },
        dt_build_submap);

    // publish ROS2 message
    double dt_to_ros2_msg = 0.0;
    auto pub_msgs = time_utils::measure_execution(
        [&]() {
            auto preprocessed_msg = toROS2msg(*this->preprocessed_pc_, msg->header);

            auto submap_msg = toROS2msg(*this->submap_pc_, msg->header);
            submap_msg->header.frame_id = "odom";

            return std::make_tuple(std::move(preprocessed_msg), std::move(submap_msg));
        },
        dt_to_ros2_msg);

    double dt_publish = 0.0;
    time_utils::measure_execution(
        [&]() {
            this->publish_odom(msg->header, this->odom_);
            if (this->pub_preprocessed_->get_subscription_count() > 0) {
                this->pub_preprocessed_->publish(std::move(std::get<0>(pub_msgs)));
            }
            if (this->pub_submap_->get_subscription_count() > 0) {
                this->pub_submap_->publish(std::move(std::get<1>(pub_msgs)));
            }
        },
        dt_publish);

    RCLCPP_INFO(this->get_logger(), "fromROS2msg:         %9.2f us", dt_from_ros2_msg);
    RCLCPP_INFO(this->get_logger(), "Preprocessing    :   %9.2f us", dt_preprocessing);
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