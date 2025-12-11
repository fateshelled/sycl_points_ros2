#include "sycl_points_ros2/lidar_odometry.hpp"

#include <rclcpp_components/register_node_macro.hpp>
#include <sycl_points/algorithms/color_gradient.hpp>
#include <sycl_points/algorithms/covariance.hpp>
#include <sycl_points/algorithms/deskew/relative_pose_deskew.hpp>
#include <sycl_points/ros2/convert.hpp>
#include <sycl_points/utils/time_utils.hpp>

namespace sycl_points {
namespace ros2 {

/// @brief constructor
/// @param options node option
LiDAROdometryNode::LiDAROdometryNode(const rclcpp::NodeOptions& options) : rclcpp::Node("lidar_odometry", options) {
    // get parameters
    this->params_ = this->get_parameters();

    // SYCL queue
    {
        // const auto device_selector = sycl_utils::device_selector::default_selector_v;
        // sycl::device dev(device_selector);
        const auto dev = sycl_utils::device_selector::select_device(this->params_.sycl_device_vendor,
                                                                    this->params_.sycl_device_type);
        this->queue_ptr_ = std::make_shared<sycl_utils::DeviceQueue>(dev);
        this->queue_ptr_->print_device_info();
    }

    // initialize buffer
    {
        this->msg_data_buffer_.reset(new shared_vector<uint8_t>(*this->queue_ptr_->ptr));
        this->scan_pc_.reset(new PointCloudShared(*this->queue_ptr_));
        this->preprocessed_pc_.reset(new PointCloudShared(*this->queue_ptr_));
        this->registration_input_pc_.reset(new PointCloudShared(*this->queue_ptr_));
        this->keyframe_pc_.reset(new PointCloudShared(*this->queue_ptr_));
        this->submap_pc_tmp_.reset(new PointCloudShared(*this->queue_ptr_));
    }

    // set Initial pose
    {
        this->odom_ = this->params_.initial_pose;
        this->prev_odom_ = this->params_.initial_pose;
    }

    // initialize keyframe
    {
        this->last_keyframe_pose_ = this->params_.initial_pose;
        this->last_keyframe_time_ = -1.0;
    }

    // Point cloud processor
    {
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
                this->params_.scan_downsampling_polar_elevation_size,
                this->params_.scan_downsampling_polar_azimuth_size, coord_system);
        }
    }

    // Submapping
    {
        if (this->params_.occupancy_grid_map_enable) {
            this->occupancy_grid_ = std::make_shared<algorithms::mapping::OccupancyGridMap>(
                *this->queue_ptr_, this->params_.submap_voxel_size);

            this->occupancy_grid_->set_log_odds_hit(this->params_.occupancy_grid_map_log_odds_hit);
            this->occupancy_grid_->set_log_odds_miss(this->params_.occupancy_grid_map_log_odds_miss);
            this->occupancy_grid_->set_log_odds_limits(this->params_.occupancy_grid_map_log_odds_limits_min,
                                                       this->params_.occupancy_grid_map_log_odds_limits_max);
            this->occupancy_grid_->set_occupancy_threshold(this->params_.occupancy_grid_map_occupied_threshold);
            this->occupancy_grid_->set_voxel_pruning_enabled(this->params_.occupancy_grid_map_enable_pruning);
            this->occupancy_grid_->set_stale_frame_threshold(this->params_.occupancy_grid_map_stale_frame_threshold);
        } else {
            this->submap_voxel_ =
                std::make_shared<algorithms::mapping::VoxelHashMap>(*this->queue_ptr_, this->params_.submap_voxel_size);
        }
    }
    // Registration
    {
        this->registration_ =
            std::make_shared<algorithms::registration::Registration>(*this->queue_ptr_, this->params_.reg_params);
    }

    // Subscription
    {
        this->sub_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "points", rclcpp::QoS(10),
            std::bind(&LiDAROdometryNode::point_cloud_callback, this, std::placeholders::_1));
    }

    // Publisher
    {
        this->pub_preprocessed_ =
            this->create_publisher<sensor_msgs::msg::PointCloud2>("sycl_lo/preprocessed", rclcpp::QoS(5));

        this->pub_submap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("sycl_lo/submap", rclcpp::QoS(5));

        this->pub_odom_ = this->create_publisher<nav_msgs::msg::Odometry>("sycl_lo/odom", rclcpp::QoS(5));
        this->pub_pose_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("sycl_lo/pose", rclcpp::QoS(5));
        this->pub_keyframe_pose_ =
            this->create_publisher<nav_msgs::msg::Odometry>("sycl_lo/keyframe/pose", rclcpp::QoS(5));

        this->tf_broadcaster_ =
            std::make_unique<tf2_ros::TransformBroadcaster>(*this, tf2_ros::DynamicBroadcasterQoS(1000));
    }

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

    // SYCL
    {
        params.sycl_device_vendor =
            this->declare_parameter<std::string>("sycl/device_vendor", params.sycl_device_vendor);
        params.sycl_device_type = this->declare_parameter<std::string>("sycl/device_type", params.sycl_device_type);
    }

    // scan
    {
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
        params.scan_downsampling_random_enable =
            this->declare_parameter<bool>("scan/downsampling/random/enable", params.scan_downsampling_random_enable);
        params.scan_downsampling_random_num =
            this->declare_parameter<int>("scan/downsampling/random/num", params.scan_downsampling_random_num);

        params.scan_covariance_neighbor_num =
            this->declare_parameter<int>("scan/covariance/neighbor_num", params.scan_covariance_neighbor_num);
        params.scan_preprocess_box_filter_enable = this->declare_parameter<bool>(
            "scan/preprocess/box_filter/enable", params.scan_preprocess_box_filter_enable);
        params.scan_preprocess_box_filter_min =
            this->declare_parameter<double>("scan/preprocess/box_filter/min", params.scan_preprocess_box_filter_min);
        params.scan_preprocess_box_filter_max =
            this->declare_parameter<double>("scan/preprocess/box_filter/max", params.scan_preprocess_box_filter_max);
    }

    // submapping
    {
        params.submap_voxel_size = this->declare_parameter<double>("submap/voxel_size", params.submap_voxel_size);
        params.submap_covariance_neighbor_num =
            this->declare_parameter<int>("submap/covariance/neighbor_num", params.submap_covariance_neighbor_num);
        params.submap_color_gradient_neighbor_num = this->declare_parameter<int>(
            "submap/color_gradient/neighbor_num", params.submap_color_gradient_neighbor_num);
        params.submap_max_distance_range =
            this->declare_parameter<double>("submap/max_distance_range", params.submap_max_distance_range);
        params.submap_point_random_sampling_num =
            this->declare_parameter<int>("submap/point_random_sampling_num", params.submap_point_random_sampling_num);

        params.keyframe_inlier_ratio_threshold = this->declare_parameter<double>(
            "submap/keyframe/inlier_ratio_threshold", params.keyframe_inlier_ratio_threshold);
        params.keyframe_distance_threshold =
            this->declare_parameter<double>("submap/keyframe/distance_threshold", params.keyframe_distance_threshold);
        params.keyframe_angle_threshold_degrees = this->declare_parameter<double>(
            "submap/keyframe/angle_threshold_degrees", params.keyframe_angle_threshold_degrees);
        params.keyframe_time_threshold_seconds = this->declare_parameter<double>(
            "submap/keyframe/time_threshold_seconds", params.keyframe_time_threshold_seconds);

        params.occupancy_grid_map_enable =
            this->declare_parameter<bool>("submap/occupancy_grid_map/enable", params.occupancy_grid_map_enable);
        params.occupancy_grid_map_log_odds_hit = this->declare_parameter<double>(
            "submap/occupancy_grid_map/log_odds_hit", params.occupancy_grid_map_log_odds_hit);
        params.occupancy_grid_map_log_odds_miss = this->declare_parameter<double>(
            "submap/occupancy_grid_map/log_odds_miss", params.occupancy_grid_map_log_odds_miss);
        params.occupancy_grid_map_log_odds_limits_min = this->declare_parameter<double>(
            "submap/occupancy_grid_map/log_odds_limits/min", params.occupancy_grid_map_log_odds_limits_min);
        params.occupancy_grid_map_log_odds_limits_max = this->declare_parameter<double>(
            "submap/occupancy_grid_map/log_odds_limits/max", params.occupancy_grid_map_log_odds_limits_max);
        params.occupancy_grid_map_occupied_threshold = this->declare_parameter<double>(
            "submap/occupancy_grid_map/occupied_threshold", params.occupancy_grid_map_occupied_threshold);
        params.occupancy_grid_map_enable_pruning = this->declare_parameter<bool>(
            "submap/occupancy_grid_map/enable_pruning", params.occupancy_grid_map_enable_pruning);
        params.occupancy_grid_map_stale_frame_threshold = this->declare_parameter<int>(
            "submap/occupancy_grid_map/stale_frame_threshold", params.occupancy_grid_map_stale_frame_threshold);
    }

    // Registration
    {
        // common
        {
            params.registration_motion_prediction_factor = this->declare_parameter<double>(
                "registration/motion_prediction_factor", params.registration_motion_prediction_factor);
            params.registration_min_num_points =
                this->declare_parameter<int>("registration/min_num_points", params.registration_min_num_points);
            params.registration_random_sampling_enable = this->declare_parameter<bool>(
                "registration/random_sampling/enable", params.registration_random_sampling_enable);
            params.registration_random_sampling_num = this->declare_parameter<int>(
                "registration/random_sampling/num", params.registration_random_sampling_num);

            const std::string reg_type = this->declare_parameter<std::string>("registration/type", "gicp");
            params.reg_params.reg_type = algorithms::registration::RegType_from_string(reg_type);
            params.reg_params.max_iterations =
                this->declare_parameter<int>("registration/max_iterations", params.reg_params.max_iterations);
            params.reg_params.lambda = this->declare_parameter<double>("registration/lambda", params.reg_params.lambda);
            params.reg_params.max_correspondence_distance = this->declare_parameter<double>(
                "registration/max_correspondence_distance", params.reg_params.max_correspondence_distance);
            params.reg_params.crireria.translation = this->declare_parameter<double>(
                "registration/criteria/translation", params.reg_params.crireria.translation);
            params.reg_params.crireria.rotation =
                this->declare_parameter<double>("registration/criteria/rotation", params.reg_params.crireria.rotation);

            params.reg_params.verbose =
                this->declare_parameter<bool>("registration/verbose", params.reg_params.verbose);
        }

        // robust
        {
            const std::string robust_loss = this->declare_parameter<std::string>("registration/robust/type", "NONE");
            params.reg_params.robust.type = algorithms::registration::RobustLossType_from_string(robust_loss);
            params.reg_params.robust.auto_scale =
                this->declare_parameter<bool>("registration/robust/auto_scale", params.reg_params.robust.auto_scale);
            params.reg_params.robust.init_scale =
                this->declare_parameter<double>("registration/robust/init_scale", params.reg_params.robust.init_scale);
            params.reg_params.robust.min_scale =
                this->declare_parameter<double>("registration/robust/min_scale", params.reg_params.robust.min_scale);
            params.reg_params.robust.scaling_iter =
                this->declare_parameter<int>("registration/robust/scaling_iter", params.reg_params.robust.scaling_iter);
        }
        // deskew
        {
            params.registration_velocity_update_enable = this->declare_parameter<bool>(
                "registration/velocity_update/enable", params.registration_velocity_update_enable);
            params.registration_velocity_update_iter = this->declare_parameter<int>(
                "registration/velocity_update/iter", params.registration_velocity_update_iter);
        }
        // photometric
        {
            params.reg_params.photometric.enable =
                this->declare_parameter<bool>("registration/photometric/enable", params.reg_params.photometric.enable);
            params.reg_params.photometric.photometric_weight = this->declare_parameter<double>(
                "registration/photometric/weight", params.reg_params.photometric.photometric_weight);
        }
        // GenZ
        {
            params.reg_params.genz.planarity_threshold = this->declare_parameter<double>(
                "registration/genz/planarity_threshold", params.reg_params.genz.planarity_threshold);
        }

        // optimization
        {
            const std::string optimization_method =
                this->declare_parameter<std::string>("registration/optimization_method", "GN");
            params.reg_params.optimization_method =
                algorithms::registration::OptimizationMethod_from_string(optimization_method);

            params.reg_params.lm.max_inner_iterations = this->declare_parameter<int>(
                "registration/lm/max_inner_iterations", params.reg_params.lm.max_inner_iterations);
            params.reg_params.lm.lambda_factor =
                this->declare_parameter<double>("registration/lm/lambda_factor", params.reg_params.lm.lambda_factor);
            params.reg_params.lm.max_lambda =
                this->declare_parameter<double>("registration/lm/max_lambda", params.reg_params.lm.max_lambda);
            params.reg_params.lm.min_lambda =
                this->declare_parameter<double>("registration/lm/min_lambda", params.reg_params.lm.min_lambda);

            params.reg_params.dogleg.initial_trust_region_radius =
                this->declare_parameter<double>("registration/dogleg/initial_trust_region_radius",
                                                params.reg_params.dogleg.initial_trust_region_radius);
            params.reg_params.dogleg.max_trust_region_radius = this->declare_parameter<double>(
                "registration/dogleg/max_trust_region_radius", params.reg_params.dogleg.max_trust_region_radius);
            params.reg_params.dogleg.min_trust_region_radius = this->declare_parameter<double>(
                "registration/dogleg/min_trust_region_radius", params.reg_params.dogleg.min_trust_region_radius);
            params.reg_params.dogleg.eta1 =
                this->declare_parameter<double>("registration/dogleg/eta1", params.reg_params.dogleg.eta1);
            params.reg_params.dogleg.eta2 =
                this->declare_parameter<double>("registration/dogleg/eta2", params.reg_params.dogleg.eta2);
            params.reg_params.dogleg.gamma_decrease = this->declare_parameter<double>(
                "registration/dogleg/gamma_decrease", params.reg_params.dogleg.gamma_decrease);
            params.reg_params.dogleg.gamma_increase = this->declare_parameter<double>(
                "registration/dogleg/gamma_increase", params.reg_params.dogleg.gamma_increase);
        }
    }

    // tf and pose
    {
        params.odom_frame_id = this->declare_parameter<std::string>("odom_frame_id", "odom");
        params.base_link_id = this->declare_parameter<std::string>("base_link_id", "base_link");
        {
            // x, y, z, qx, qy, qz, qw
            const auto x = this->declare_parameter<double>("T_base_link_to_lidar/x", 0.0);
            const auto y = this->declare_parameter<double>("T_base_link_to_lidar/y", 0.0);
            const auto z = this->declare_parameter<double>("T_base_link_to_lidar/z", 0.0);
            const auto qx = this->declare_parameter<double>("T_base_link_to_lidar/qx", 0.0);
            const auto qy = this->declare_parameter<double>("T_base_link_to_lidar/qy", 0.0);
            const auto qz = this->declare_parameter<double>("T_base_link_to_lidar/qz", 0.0);
            const auto qw = this->declare_parameter<double>("T_base_link_to_lidar/qw", 1.0);
            params.T_base_link_to_lidar.setIdentity();
            params.T_base_link_to_lidar.translation() << x, y, z;
            const Eigen::Quaternionf quat(qw, qx, qy, qz);
            params.T_base_link_to_lidar.matrix().block<3, 3>(0, 0) = quat.matrix();

            params.T_lidar_to_base_link = params.T_base_link_to_lidar.inverse();
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
    }
    return params;
}

void LiDAROdometryNode::point_cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
    const double timestamp = rclcpp::Time(msg->header.stamp).seconds();
    if (this->last_frame_time_ > 0.0) {
        this->dt_ = static_cast<float>(timestamp - this->last_frame_time_);
    }

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

            if (this->params_.scan_downsampling_random_enable) {
                preprocess_filter_->random_sampling(*this->preprocessed_pc_, *this->preprocessed_pc_,
                                                    this->params_.scan_downsampling_random_num);
            }
        },
        dt_preprocessing);

    if (this->preprocessed_pc_->size() <= this->params_.registration_min_num_points) {
        RCLCPP_WARN(this->get_logger(), "point cloud size is too small");
        return;
    }

    // compute covariances
    double dt_covariance = 0.0;
    time_utils::measure_execution(
        [&]() {
            if (this->params_.reg_params.reg_type == algorithms::registration::RegType::GICP) {
                // build KDTree
                const auto src_tree = algorithms::knn::KDTree::build(*this->queue_ptr_, *this->preprocessed_pc_);
                auto events = src_tree->knn_search_async(*this->preprocessed_pc_,
                                                         this->params_.scan_covariance_neighbor_num, this->knn_result_);
                events += algorithms::covariance::compute_covariances_async(this->knn_result_, *this->preprocessed_pc_,
                                                                            events.evs);
                // events += algorithms::covariance::covariance_update_plane_async(*this->preprocessed_pc_,
                // events.evs);
                events += algorithms::covariance::covariance_normalize_async(*this->preprocessed_pc_, events.evs);
                events.wait_and_throw();
            }
        },
        dt_covariance);

    auto build_submap = [&](const PointCloudShared::Ptr& pc, const Eigen::Isometry3f& current_pose) {
        // random sampling
        preprocess_filter_->random_sampling(*pc, *this->keyframe_pc_, this->params_.submap_point_random_sampling_num);

        // mapping
        if (this->params_.occupancy_grid_map_enable) {
            this->occupancy_grid_->add_point_cloud(*this->keyframe_pc_, current_pose);
            this->occupancy_grid_->extract_occupied_points(*this->submap_pc_tmp_, current_pose,
                                                           this->params_.submap_max_distance_range);
            // 360 deg x 180 deg
            // this->occupancy_grid_->extract_visible_points(  //
            //     *this->submap_pc_tmp_, current_pose, 30.0f, 2.0f * M_PIf, M_PIf);
        } else {
            this->submap_voxel_->add_point_cloud(*this->keyframe_pc_, current_pose);
            this->submap_voxel_->downsampling(*this->submap_pc_tmp_, current_pose.translation(),
                                              this->params_.submap_max_distance_range);
        }

        if (this->is_first_frame_) {
            // deep copy
            this->submap_pc_ptr_.reset(new PointCloudShared(*this->queue_ptr_, *pc));
        } else if (this->submap_pc_tmp_->size() >= this->params_.registration_min_num_points) {
            // copy pointer
            this->submap_pc_ptr_ = this->submap_pc_tmp_;
        }

        this->submap_tree_ = algorithms::knn::KDTree::build(*this->queue_ptr_, *this->submap_pc_ptr_);

        auto knn_events = this->submap_tree_->knn_search_async(
            *this->submap_pc_ptr_, this->params_.submap_covariance_neighbor_num, this->knn_result_);

        // compute grad
        sycl_utils::events grad_events;
        if (this->params_.reg_params.photometric.enable) {
            if (this->submap_pc_ptr_->has_rgb()) {
                if (this->params_.submap_covariance_neighbor_num != this->params_.submap_color_gradient_neighbor_num) {
                    grad_events += this->submap_tree_->knn_search_async(
                        *this->submap_pc_ptr_, this->params_.submap_color_gradient_neighbor_num,
                        this->knn_result_grad_);
                    grad_events += algorithms::color_gradient::compute_color_gradients_async(
                        *this->submap_pc_ptr_, this->knn_result_grad_, grad_events.evs);
                } else {
                    grad_events += algorithms::color_gradient::compute_color_gradients_async(
                        *this->submap_pc_ptr_, this->knn_result_, knn_events.evs);
                }
            } else if (this->submap_pc_ptr_->has_intensity()) {
                if (this->params_.submap_covariance_neighbor_num != this->params_.submap_color_gradient_neighbor_num) {
                    grad_events += this->submap_tree_->knn_search_async(
                        *this->submap_pc_ptr_, this->params_.submap_color_gradient_neighbor_num,
                        this->knn_result_grad_);
                    grad_events += algorithms::intensity_gradient::compute_intensity_gradients_async(
                        *this->submap_pc_ptr_, this->knn_result_grad_, grad_events.evs);
                } else {
                    grad_events += algorithms::intensity_gradient::compute_intensity_gradients_async(
                        *this->submap_pc_ptr_, this->knn_result_, knn_events.evs);
                }
            }
        }
        // compute covariances and normals
        sycl_utils::events cov_events;
        if (this->params_.reg_params.reg_type != algorithms::registration::RegType::POINT_TO_POINT) {
            cov_events += algorithms::covariance::compute_covariances_async(this->knn_result_, *this->submap_pc_ptr_,
                                                                            knn_events.evs);

            if (this->params_.reg_params.reg_type == algorithms::registration::RegType::POINT_TO_PLANE ||
                this->params_.reg_params.reg_type == algorithms::registration::RegType::GENZ) {
                cov_events +=
                    algorithms::covariance::compute_normals_from_covariances_async(*this->submap_pc_ptr_,
                    cov_events.evs);
            }
            if (this->params_.reg_params.reg_type == algorithms::registration::RegType::GICP) {
                cov_events +=
                    algorithms::covariance::covariance_update_plane_async(*this->submap_pc_ptr_, cov_events.evs);
                // cov_events += algorithms::covariance::covariance_normalize_async(*this->submap_pc_ptr_,
                // cov_events.evs);
            }
        }
        knn_events.wait_and_throw();
        grad_events.wait_and_throw();
        cov_events.wait_and_throw();
    };

    if (this->is_first_frame_) {
        build_submap(this->preprocessed_pc_, this->params_.initial_pose);

        this->is_first_frame_ = false;
        this->last_keyframe_time_ = timestamp;
        this->last_frame_time_ = timestamp;

        return;
    }

    // Registration
    double dt_registration = 0.0;
    const auto reg_result = time_utils::measure_execution(
        [&]() {
            // Predict initial pose by applying the previous motion model
            Eigen::Isometry3f init_T = Eigen::Isometry3f::Identity();
            if (this->params_.registration_motion_prediction_factor > 0.0f &&
                this->params_.registration_motion_prediction_factor <= 1.0f) {
                const auto delta_pose = this->prev_odom_.inverse() * this->odom_;
                const Eigen::Vector3f delta_trans = delta_pose.translation();
                const Eigen::AngleAxisf delta_angle_axis(delta_pose.rotation());

                const Eigen::Vector3f predicted_trans =
                    this->odom_.translation() +
                    this->odom_.rotation() * (delta_trans * this->params_.registration_motion_prediction_factor);
                const Eigen::Quaternionf predicted_rot =
                    Eigen::AngleAxisf(delta_angle_axis.angle() * this->params_.registration_motion_prediction_factor,
                                      delta_angle_axis.axis()) *
                    Eigen::Quaternionf(this->odom_.rotation());

                init_T.translation() = predicted_trans;
                init_T.rotate(predicted_rot.normalized());
            } else {
                init_T = this->odom_;
            }

            if (this->params_.registration_random_sampling_enable) {
                this->preprocess_filter_->random_sampling(*this->preprocessed_pc_, *this->registration_input_pc_,
                                                          this->params_.registration_random_sampling_num);
            } else {
                *this->registration_input_pc_ = *this->preprocessed_pc_;
            }
            algorithms::registration::RegistrationResult result;

            if (this->params_.registration_velocity_update_enable) {
                result = this->registration_->align_velocity_update(
                    *this->registration_input_pc_, *this->submap_pc_ptr_, *this->submap_tree_, init_T.matrix(),
                    this->dt_, this->params_.registration_velocity_update_iter, this->odom_.matrix());
            } else {
                result = this->registration_->align(*this->registration_input_pc_, *this->submap_pc_ptr_,
                                                    *this->submap_tree_, init_T.matrix());
            }

            return result;
        },
        dt_registration);

    // Build submap
    double dt_build_submap = 0.0;
    const bool update_submap = time_utils::measure_execution(
        [&]() {
            if (this->params_.registration_velocity_update_enable) {
                algorithms::deskew::deskew_point_cloud_constant_velocity(
                    *this->preprocessed_pc_, *this->preprocessed_pc_, this->odom_, reg_result.T, this->dt_);
            }

            // Conditions for adding keyframes
            //   inlier_ratio > keyframe_inlier_ratio_threshold
            //   &&
            //   ( distance > keyframe_distance_threshold ||
            //     angle >= this->params_.keyframe_angle_threshold_degrees ||
            //     delta_t >= this->params_.keyframe_time_threshold_seconds )

            const float inlier_ratio = static_cast<float>(reg_result.inlier) / this->registration_input_pc_->size();
            if (inlier_ratio <= this->params_.keyframe_inlier_ratio_threshold) {
                return false;
            }

            // for octomap
            if (this->params_.occupancy_grid_map_enable) {
                build_submap(this->preprocessed_pc_, reg_result.T);
                return true;
            }

            // calculate delta pose
            const auto delta_pose = this->last_keyframe_pose_.inverse() * reg_result.T;

            // calculate moving distance and angle
            const auto distance = delta_pose.translation().norm();
            const auto angle = std::fabs(Eigen::AngleAxisf(delta_pose.rotation()).angle()) * (180.0f / M_PIf);

            // calculate delta time
            const auto delta_time = this->last_keyframe_time_ > 0.0 ? timestamp - this->last_keyframe_time_
                                                                    : std::numeric_limits<double>::max();

            // update submap
            if (distance >= this->params_.keyframe_distance_threshold ||
                angle >= this->params_.keyframe_angle_threshold_degrees ||
                delta_time >= this->params_.keyframe_time_threshold_seconds) {
                this->last_keyframe_pose_ = reg_result.T;
                this->last_keyframe_time_ = timestamp;

                build_submap(this->preprocessed_pc_, reg_result.T);
                return true;
            }
            return false;
        },
        dt_build_submap);

    // update Odometry
    {
        this->prev_odom_ = this->odom_;
        this->odom_ = reg_result.T;
    }

    // publish ROS2 message
    double dt_to_ros2_msg = 0.0;
    auto preprocessed_msg = time_utils::measure_execution(
        [&]() { return toROS2msg(*this->preprocessed_pc_, msg->header); }, dt_to_ros2_msg);
    auto submap_msg = time_utils::measure_execution(
        [&]() {
            if (update_submap) {
                auto submap_msg = toROS2msg(*this->submap_pc_ptr_, msg->header);
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
            this->publish_odom(msg->header, reg_result);
            if (preprocessed_msg != nullptr && this->pub_preprocessed_->get_subscription_count() > 0) {
                this->pub_preprocessed_->publish(*preprocessed_msg);
            }
            if (update_submap) {
                this->publish_keyframe_pose(msg->header, this->last_keyframe_pose_);
            }
            if (update_submap && this->pub_submap_->get_subscription_count() > 0) {
                this->pub_submap_->publish(*submap_msg);
            }
        },
        dt_publish);

    this->last_frame_time_ = timestamp;

    double total_time = 0.0;
    total_time += dt_from_ros2_msg;
    total_time += dt_preprocessing;
    total_time += dt_covariance;
    total_time += dt_registration;
    total_time += dt_build_submap;
    total_time += dt_to_ros2_msg;
    total_time += dt_publish;

    this->add_delta_time("1. fromROS2msg:         ", dt_from_ros2_msg);
    this->add_delta_time("2. Preprocessing:       ", dt_preprocessing);
    this->add_delta_time("3. compute Covariances: ", dt_covariance);
    this->add_delta_time("4. Registration:        ", dt_registration);
    this->add_delta_time("5. Build submap:        ", dt_build_submap);
    this->add_delta_time("6. toROS2msg:           ", dt_to_ros2_msg);
    this->add_delta_time("7. publish:             ", dt_publish);
    this->add_delta_time("8. total:               ", total_time);

    RCLCPP_INFO(this->get_logger(), "1. fromROS2msg:         %9.2f us", dt_from_ros2_msg);
    RCLCPP_INFO(this->get_logger(), "2. Preprocessing:       %9.2f us", dt_preprocessing);
    RCLCPP_INFO(this->get_logger(), "3. compute Covariances: %9.2f us", dt_covariance);
    RCLCPP_INFO(this->get_logger(), "4. Registration:        %9.2f us", dt_registration);
    RCLCPP_INFO(this->get_logger(), "5. Build submap:        %9.2f us", dt_build_submap);
    RCLCPP_INFO(this->get_logger(), "6. toROS2msg:           %9.2f us", dt_to_ros2_msg);
    RCLCPP_INFO(this->get_logger(), "7. publish:             %9.2f us", dt_publish);
    RCLCPP_INFO(this->get_logger(), "8. total:               %9.2f us", total_time);
    RCLCPP_INFO(this->get_logger(), "");
}

void LiDAROdometryNode::publish_odom(const std_msgs::msg::Header& header,
                                     const algorithms::registration::RegistrationResult& reg_result) {
    const auto odom_trans = reg_result.T.translation();
    const Eigen::Quaternionf odom_quat(reg_result.T.rotation());
    {
        geometry_msgs::msg::TransformStamped::SharedPtr tf;
        tf.reset(new geometry_msgs::msg::TransformStamped);
        tf->header.stamp = header.stamp;
        tf->header.frame_id = this->params_.odom_frame_id;
        tf->child_frame_id = this->params_.base_link_id;

        tf->transform.translation.x = odom_trans.x();
        tf->transform.translation.y = odom_trans.y();
        tf->transform.translation.z = odom_trans.z();
        tf->transform.rotation.x = odom_quat.x();
        tf->transform.rotation.y = odom_quat.y();
        tf->transform.rotation.z = odom_quat.z();
        tf->transform.rotation.w = odom_quat.w();
        this->tf_broadcaster_->sendTransform(*std::move(tf));
    }

    geometry_msgs::msg::PoseStamped pose;
    pose.header.stamp = header.stamp;
    pose.header.frame_id = this->params_.odom_frame_id;
    pose.pose.position.x = odom_trans.x();
    pose.pose.position.y = odom_trans.y();
    pose.pose.position.z = odom_trans.z();
    pose.pose.orientation.x = odom_quat.x();
    pose.pose.orientation.y = odom_quat.y();
    pose.pose.orientation.z = odom_quat.z();
    pose.pose.orientation.w = odom_quat.w();
    this->pub_pose_->publish(pose);

    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = header.stamp;
    odom_msg.header.frame_id = this->params_.odom_frame_id;
    odom_msg.child_frame_id = this->params_.base_link_id;
    odom_msg.pose.pose.position = pose.pose.position;
    odom_msg.pose.pose.orientation = pose.pose.orientation;
    // convert Hessian to Covariance
    {
        // rotation first
        const Eigen::Matrix<float, 6, 6> cov_reg = reg_result.H.inverse();

        // ROS covariance is translation first
        Eigen::Matrix<float, 6, 6> cov_odom;
        cov_odom.block<3, 3>(0, 0) = cov_reg.block<3, 3>(3, 3);
        cov_odom.block<3, 3>(3, 3) = cov_reg.block<3, 3>(0, 0);
        cov_odom.block<3, 3>(0, 3) = cov_reg.block<3, 3>(3, 0);
        cov_odom.block<3, 3>(3, 0) = cov_reg.block<3, 3>(0, 3);

        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>(odom_msg.pose.covariance.data()) =
            cov_odom.cast<double>();
    }
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
