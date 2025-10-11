#include "sycl_points_ros2/lidar_odometry.hpp"

#include <rclcpp_components/register_node_macro.hpp>
#include <sycl_points/algorithms/color_gradient.hpp>
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

    // initialize buffer
    this->msg_data_buffer_ = std::make_shared<shared_vector<uint8_t>>(*this->queue_ptr_->ptr);

    // set Initial pose
    this->odom_ = this->params_.initial_pose;
    this->last_keyframe_pose_ = this->params_.initial_pose;
    this->last_keyframe_time_ = -1.0;

    this->scan_pc_.reset(new PointCloudShared(*this->queue_ptr_));
    this->preprocessed_pc_.reset(new PointCloudShared(*this->queue_ptr_));

    // Point cloud processor
    this->preprocess_filter_ = std::make_shared<algorithms::filter::PreprocessFilter>(*this->queue_ptr_);
    if (this->params_.scan_downsampling_voxel_enable) {
        this->voxel_filter_ = std::make_shared<algorithms::filter::VoxelGrid>(
            *this->queue_ptr_, this->params_.scan_downsampling_voxel_size);
    }
    if (this->params_.scan_downsampling_polar_enable) {
        const auto coord_system =
            algorithms::coordinate_system_from_string(this->params_.scan_downsampling_polar_coord_system);
        this->polar_filter_ = std::make_shared<algorithms::filter::PolarGrid>(
            *this->queue_ptr_, this->params_.scan_downsampling_polar_distance_size,
            this->params_.scan_downsampling_polar_elevation_size, this->params_.scan_downsampling_polar_azimuth_size,
            coord_system);
    }
    this->submap_voxel_filter_ = std::make_shared<algorithms::filter::VoxelGrid>(
        *this->queue_ptr_, this->params_.submap_downsampling_voxel_size);
    this->gicp_ = std::make_shared<algorithms::registration::RegistrationGICP>(*this->queue_ptr_, this->params_.gicp);

    // pub/sub
    this->sub_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "points", rclcpp::QoS(10), std::bind(&LiDAROdometryNode::point_cloud_callback, this, std::placeholders::_1));

    this->pub_preprocessed_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>("sycl_lo/preprocessed", rclcpp::SensorDataQoS());

    auto submap_qos = rclcpp::QoS(1);
    submap_qos.durability(rclcpp::DurabilityPolicy::TransientLocal);
    this->pub_submap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("sycl_lo/submap", submap_qos);

    this->pub_odom_ = this->create_publisher<nav_msgs::msg::Odometry>("sycl_lo/odom", rclcpp::QoS(10));
    this->pub_pose_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("sycl_lo/pose", rclcpp::QoS(10));
    this->pub_keyframe_pose_ =
        this->create_publisher<nav_msgs::msg::Odometry>("sycl_lo/keyframe/pose", rclcpp::QoS(10));

    this->tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    RCLCPP_INFO(this->get_logger(), "Subscribe PointCloud: %s", this->sub_pc_->get_topic_name());
    RCLCPP_INFO(this->get_logger(), "Publish Preprocessed PointCloud: %s", this->pub_preprocessed_->get_topic_name());
    RCLCPP_INFO(this->get_logger(), "Publish Submap PointCloud: %s", this->pub_submap_->get_topic_name());
    RCLCPP_INFO(this->get_logger(), "Publish Odometry: %s", this->pub_odom_->get_topic_name());
    RCLCPP_INFO(this->get_logger(), "Publish Pose: %s", this->pub_pose_->get_topic_name());
    RCLCPP_INFO(this->get_logger(), "Publish Keyframe Pose: %s", this->pub_keyframe_pose_->get_topic_name());
}

/// @brief destructor
LiDAROdometryNode::~LiDAROdometryNode() {
    // processing time log
    RCLCPP_INFO(this->get_logger(), "");
    RCLCPP_INFO(this->get_logger(), "MAX processing time");
    for (auto& item : this->processing_times_) {
        const double max = *std::max_element(item.second.begin(), item.second.end());
        RCLCPP_INFO(this->get_logger(), "%s %9.2f us", item.first.c_str(), max);
    }

    RCLCPP_INFO(this->get_logger(), "");
    RCLCPP_INFO(this->get_logger(), "MEAN processing time");
    for (auto& item : this->processing_times_) {
        const double avg =
            std::accumulate(item.second.begin(), item.second.end(), 0.0) / static_cast<double>(item.second.size());
        RCLCPP_INFO(this->get_logger(), "%s %9.2f us", item.first.c_str(), avg);
    }

    RCLCPP_INFO(this->get_logger(), "");
    RCLCPP_INFO(this->get_logger(), "MEDIAN processing time");
    for (auto& item : this->processing_times_) {
        std::sort(item.second.begin(), item.second.end());
        const double median = item.second[item.second.size() / 2];
        RCLCPP_INFO(this->get_logger(), "%s %9.2f us", item.first.c_str(), median);
    }
    RCLCPP_INFO(this->get_logger(), "");
}

LiDAROdometryNode::Parameters LiDAROdometryNode::get_parameters() {
    LiDAROdometryNode::Parameters params;

    params.sycl_device_vendor = this->declare_parameter<std::string>("sycl/device_vendor", params.sycl_device_vendor);
    params.sycl_device_type = this->declare_parameter<std::string>("sycl/device_type", params.sycl_device_type);

    params.scan_downsampling_voxel_enable =
        this->declare_parameter<bool>("scan/downsampling/voxel/enable", params.scan_downsampling_voxel_enable);
    params.scan_downsampling_voxel_size =
        this->declare_parameter<double>("scan/downsampling/voxel/voxel_size", params.scan_downsampling_voxel_size);

    params.scan_downsampling_polar_enable =
        this->declare_parameter<bool>("scan/downsampling/polar/enable", params.scan_downsampling_polar_enable);
    params.scan_downsampling_polar_distance_size = this->declare_parameter<double>(
        "scan/downsampling/polar/distance_size", params.scan_downsampling_polar_distance_size);
    params.scan_downsampling_polar_elevation_size = this->declare_parameter<double>(
        "scan/downsampling/polar/elevation_size", params.scan_downsampling_polar_elevation_size);
    params.scan_downsampling_polar_azimuth_size = this->declare_parameter<double>(
        "scan/downsampling/polar/azimuth_size", params.scan_downsampling_polar_azimuth_size);
    params.scan_downsampling_polar_coord_system = this->declare_parameter<std::string>(
        "scan/downsampling/polar/coord_system", params.scan_downsampling_polar_coord_system);

    params.scan_covariance_neighbor_num =
        this->declare_parameter<int>("scan/covariance/neighbor_num", params.scan_covariance_neighbor_num);
    params.scan_preprocess_box_filter_enable =
        this->declare_parameter<bool>("scan/preprocess/box_filter/enable", params.scan_preprocess_box_filter_enable);
    params.scan_preprocess_box_filter_min =
        this->declare_parameter<double>("scan/preprocess/box_filter/min", params.scan_preprocess_box_filter_min);
    params.scan_preprocess_box_filter_max =
        this->declare_parameter<double>("scan/preprocess/box_filter/max", params.scan_preprocess_box_filter_max);
    params.scan_preprocess_random_sampling_enable = this->declare_parameter<bool>(
        "scan/preprocess/random_sampling/enable", params.scan_preprocess_random_sampling_enable);
    params.scan_preprocess_random_sampling_num =
        this->declare_parameter<int>("scan/preprocess/random_sampling/num", params.scan_preprocess_random_sampling_num);

    params.submap_downsampling_voxel_size =
        this->declare_parameter<double>("submap/downsampling/voxel/voxel_size", params.submap_downsampling_voxel_size);
    params.submap_covariance_neighbor_num =
        this->declare_parameter<int>("submap/covariance/neighbor_num", params.submap_covariance_neighbor_num);
    params.submap_color_gradient_neighbor_num =
        this->declare_parameter<int>("submap/color_gradient/neighbor_num", params.submap_color_gradient_neighbor_num);

    params.keyframe_inlier_ratio_threshold =
        this->declare_parameter<double>("keyframe/inlier_ratio_threshold", params.keyframe_inlier_ratio_threshold);
    params.keyframe_distance_threshold =
        this->declare_parameter<double>("keyframe/distance_threshold", params.keyframe_distance_threshold);
    params.keyframe_angle_threshold_degrees =
        this->declare_parameter<double>("keyframe/angle_threshold_degrees", params.keyframe_angle_threshold_degrees);
    params.keyframe_time_threshold_seconds =
        this->declare_parameter<double>("keyframe/time_threshold_seconds", params.keyframe_time_threshold_seconds);

    params.gicp_min_num_points = this->declare_parameter<int>("gicp/min_num_points", 100);

    params.gicp.max_iterations = this->declare_parameter<int>("gicp/max_iterations", 20);
    params.gicp.lambda = this->declare_parameter<double>("gicp/lambda", 1e-4);
    params.gicp.max_correspondence_distance = this->declare_parameter<double>("gicp/max_correspondence_distance", 2.0);
    params.gicp.crireria.translation = this->declare_parameter<double>("gicp/crireria/translation", 1e-3);
    params.gicp.crireria.rotation = this->declare_parameter<double>("gicp/crireria/rotation", 1e-3);

    const std::string robust_loss = this->declare_parameter<std::string>("gicp/robust/type", "NONE");
    params.gicp.robust.type = algorithms::registration::RobustLossType_from_string(robust_loss);
    params.gicp.robust.scale = this->declare_parameter<double>("gicp/robust/scale", 1.0);
    params.gicp.photometric.enable = this->declare_parameter<bool>("gicp/photometric/enable", false);
    params.gicp.photometric.photometric_weight = this->declare_parameter<double>("gicp/photometric/weight", 0.0f);

    params.gicp.lm.enable = this->declare_parameter<bool>("gicp/lm/enable", false);
    params.gicp.lm.max_inner_iterations = this->declare_parameter<int>("gicp/lm/max_inner_iterations", 10);
    params.gicp.lm.lambda_factor = this->declare_parameter<double>("gicp/lm/lambda_factor", 10.0);

    params.gicp.verbose = this->declare_parameter<bool>("gicp/verbose", true);

    params.odom_frame_id = this->declare_parameter<std::string>("odom_frame_id", "odom");
    params.base_link_id = this->declare_parameter<std::string>("base_link_id", "base_link");
    {
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
    const double timestamp = rclcpp::Time(msg->header.stamp).seconds();
    const bool is_first_frame = (this->submap_pc_ == nullptr);

    double dt_from_ros2_msg = 0.0;
    time_utils::measure_execution(
        [&]() { return fromROS2msg(*this->queue_ptr_, *msg, this->scan_pc_, this->msg_data_buffer_); },
        dt_from_ros2_msg);

    if (this->scan_pc_->size() == 0) {
        RCLCPP_WARN(this->get_logger(), "input point cloud is empty");
        return;
    }

    // preprocess
    double dt_preprocessing = 0.0;
    time_utils::measure_execution(
        [&]() {
            // box filter -> polar grid -> voxel grid
            if (this->params_.scan_preprocess_box_filter_enable) {
                this->preprocess_filter_->box_filter(*this->scan_pc_, this->params_.scan_preprocess_box_filter_min,
                                                     this->params_.scan_preprocess_box_filter_max);
            }
            if (this->params_.scan_downsampling_polar_enable) {
                this->polar_filter_->downsampling(*this->scan_pc_, *this->preprocessed_pc_);
                if (this->params_.scan_downsampling_voxel_enable) {
                    this->voxel_filter_->downsampling(*this->preprocessed_pc_, *this->preprocessed_pc_);
                }
            } else {
                if (this->params_.scan_downsampling_voxel_enable) {
                    this->voxel_filter_->downsampling(*this->scan_pc_, *this->preprocessed_pc_);
                } else {
                    *this->preprocessed_pc_ = *this->scan_pc_;  // copy
                }
            }
            if (is_first_frame) {
                sycl_points::algorithms::transform::transform(*this->preprocessed_pc_,
                                                              this->params_.initial_pose.matrix());
            }
        },
        dt_preprocessing);

    if (this->preprocessed_pc_->size() <= this->params_.gicp_min_num_points) {
        RCLCPP_WARN(this->get_logger(), "point cloud size is too small");
        return;
    }

    // build KDTree
    double dt_kdtree_build = 0.0;
    const auto src_tree = time_utils::measure_execution(
        [&]() {
            const auto tree =
                sycl_points::algorithms::knn_search::KDTree::build(*this->queue_ptr_, *this->preprocessed_pc_);
            return tree;
        },
        dt_kdtree_build);

    // compute covariances
    double dt_covariance = 0.0;
    time_utils::measure_execution(
        [&]() {
            src_tree
                ->knn_search_async(*this->preprocessed_pc_, this->params_.scan_covariance_neighbor_num,
                                   this->knn_result_)
                .wait();
            algorithms::covariance::compute_covariances_async(this->knn_result_, *this->preprocessed_pc_).wait();
            algorithms::covariance::compute_normals_from_covariances_async(*this->preprocessed_pc_).wait();
            algorithms::covariance::covariance_update_plane(*this->preprocessed_pc_);
        },
        dt_covariance);

    // is first frame
    if (is_first_frame) {
        // copy to submap
        this->submap_pc_ = std::make_shared<PointCloudShared>(*this->preprocessed_pc_);
        this->submap_tree_ = src_tree;
        if (this->submap_pc_->has_rgb() && this->params_.gicp.photometric.enable) {
            algorithms::color_gradient::compute_color_gradients_async(*this->submap_pc_, this->knn_result_).wait();
        }
        this->last_keyframe_time_ = timestamp;
        return;
    }

    // Registration
    double dt_registration = 0.0;
    const auto reg_result = time_utils::measure_execution(
        [&]() {
            const Eigen::Isometry3f init_T = this->odom_;

            if (this->params_.scan_preprocess_random_sampling_enable) {
                this->preprocess_filter_->random_sampling(*this->preprocessed_pc_,
                                                          this->params_.scan_preprocess_random_sampling_num);
            }

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
            // Conditions for adding keyframes
            //   inlier_ratio > keyframe_inlier_ratio_threshold
            //   &&
            //   ( distance > keyframe_distance_threshold ||
            //     angle >= this->params_.keyframe_angle_threshold_degrees ||
            //     delta_t >= this->params_.keyframe_time_threshold_seconds )

            const float inlier_ratio = static_cast<float>(reg_result.inlier) / this->preprocessed_pc_->size();
            if (inlier_ratio <= this->params_.keyframe_inlier_ratio_threshold) {
                return;
            }

            // calculate delta pose
            const auto delta_pose = this->last_keyframe_pose_.inverse() * reg_result.T;

            // calculate moving distance and angle
            const auto distance = delta_pose.translation().norm();
            const auto angle = Eigen::AngleAxisf(delta_pose.rotation()).angle() * (180.0f / M_PIf);

            // calculate delta time
            const auto delta_time = this->last_keyframe_time_ > 0.0 ? timestamp - this->last_keyframe_time_
                                                                    : std::numeric_limits<double>::lowest();

            // update submap
            if (distance >= this->params_.keyframe_distance_threshold ||
                angle >= this->params_.keyframe_angle_threshold_degrees ||
                delta_time >= this->params_.keyframe_time_threshold_seconds) {
                this->last_keyframe_pose_ = reg_result.T;
                this->last_keyframe_time_ = timestamp;

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

                this->submap_tree_
                    ->knn_search_async(*this->submap_pc_, this->params_.submap_covariance_neighbor_num,
                                       this->knn_result_)
                    .wait();

                algorithms::covariance::compute_covariances_async(this->knn_result_, *this->submap_pc_).wait();
                algorithms::covariance::compute_normals_from_covariances_async(*this->submap_pc_).wait();
                algorithms::covariance::covariance_update_plane(*this->submap_pc_);

                if (this->submap_pc_->has_rgb() && this->params_.gicp.photometric.enable) {
                    if (this->params_.submap_covariance_neighbor_num !=
                        this->params_.submap_color_gradient_neighbor_num) {
                        this->submap_tree_
                            ->knn_search_async(*this->submap_pc_, this->params_.submap_color_gradient_neighbor_num,
                                               this->knn_result_)
                            .wait();
                    }
                    algorithms::color_gradient::compute_color_gradients_async(*this->submap_pc_, this->knn_result_)
                        .wait();
                }
                update_submap = true;
            }
        },
        dt_build_submap);

    // publish ROS2 message
    double dt_to_ros2_msg = 0.0;
    auto preprocessed_msg = time_utils::measure_execution(
        [&]() { return toROS2msg(*this->preprocessed_pc_, msg->header); }, dt_to_ros2_msg);
    auto submap_msg = time_utils::measure_execution(
        [&]() {
            if (true) {
                auto submap_msg = toROS2msg(*this->submap_pc_, msg->header);
                submap_msg->header.frame_id = this->params_.odom_frame_id;
                return submap_msg;
            }
            sensor_msgs::msg::PointCloud2::SharedPtr ret = nullptr;
            return ret;
        },
        dt_to_ros2_msg);

    double dt_publish = 0.0;
    time_utils::measure_execution(
        [&]() {
            this->publish_odom(msg->header, this->odom_);
            if (preprocessed_msg != nullptr && this->pub_preprocessed_->get_subscription_count() > 0) {
                this->pub_preprocessed_->publish(*preprocessed_msg);
            }
            if (update_submap) {
                this->publish_keyframe_pose(msg->header, this->last_keyframe_pose_);
            }
            if (submap_msg != nullptr && this->pub_submap_->get_subscription_count() > 0) {
                this->pub_submap_->publish(*submap_msg);
            }
        },
        dt_publish);

    double total_time = 0.0;
    total_time += dt_from_ros2_msg;
    total_time += dt_preprocessing;
    total_time += dt_kdtree_build;
    total_time += dt_covariance;
    total_time += dt_registration;
    total_time += dt_build_submap;
    total_time += dt_to_ros2_msg;
    total_time += dt_publish;

    this->add_delta_time("1. fromROS2msg:         ", dt_from_ros2_msg);
    this->add_delta_time("2. Preprocessing:       ", dt_preprocessing);
    this->add_delta_time("3. KDTree build:        ", dt_kdtree_build);
    this->add_delta_time("4. compute Covariances: ", dt_covariance);
    this->add_delta_time("5. Registration:        ", dt_registration);
    this->add_delta_time("6. Build submap:        ", dt_build_submap);
    this->add_delta_time("7. toROS2msg:           ", dt_to_ros2_msg);
    this->add_delta_time("8. publish:             ", dt_publish);
    this->add_delta_time("9. total:               ", total_time);

    RCLCPP_INFO(this->get_logger(), "1. fromROS2msg:         %9.2f us", dt_from_ros2_msg);
    RCLCPP_INFO(this->get_logger(), "2. Preprocessing:       %9.2f us", dt_preprocessing);
    RCLCPP_INFO(this->get_logger(), "3. KDTree build:        %9.2f us", dt_kdtree_build);
    RCLCPP_INFO(this->get_logger(), "4. compute Covariances: %9.2f us", dt_covariance);
    RCLCPP_INFO(this->get_logger(), "5. Registration:        %9.2f us", dt_registration);
    RCLCPP_INFO(this->get_logger(), "6. Build submap:        %9.2f us", dt_build_submap);
    RCLCPP_INFO(this->get_logger(), "7. toROS2msg:           %9.2f us", dt_to_ros2_msg);
    RCLCPP_INFO(this->get_logger(), "8. publish:             %9.2f us", dt_publish);
    RCLCPP_INFO(this->get_logger(), "9. total:               %9.2f us", total_time);
    RCLCPP_INFO(this->get_logger(), "");
}

void LiDAROdometryNode::publish_odom(const std_msgs::msg::Header& header, const Eigen::Isometry3f& odom) {
    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = header.stamp;
    tf.header.frame_id = this->params_.odom_frame_id;
    tf.child_frame_id = this->params_.base_link_id;

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

void LiDAROdometryNode::publish_keyframe_pose(const std_msgs::msg::Header& header, const Eigen::Isometry3f& odom) {
    const auto odom_trans = odom.translation();
    const Eigen::Quaternionf odom_quat(odom.rotation());

    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header = header;
    odom_msg.header.frame_id = this->params_.odom_frame_id;
    odom_msg.child_frame_id = this->params_.base_link_id;
    odom_msg.pose.pose.position.x = odom_trans.x();
    odom_msg.pose.pose.position.y = odom_trans.y();
    odom_msg.pose.pose.position.z = odom_trans.z();
    odom_msg.pose.pose.orientation.x = odom_quat.x();
    odom_msg.pose.pose.orientation.y = odom_quat.y();
    odom_msg.pose.pose.orientation.z = odom_quat.z();
    odom_msg.pose.pose.orientation.w = odom_quat.w();
    this->pub_keyframe_pose_->publish(odom_msg);
}
}  // namespace ros2
}  // namespace sycl_points

RCLCPP_COMPONENTS_REGISTER_NODE(sycl_points::ros2::LiDAROdometryNode)
