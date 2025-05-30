#pragma once

#include <Eigen/Dense>
#include <cmath>

namespace sycl_points {
namespace ekf {

class PoseEKF {
public:
    static constexpr int STATE_DIM = 13;  // pos(3) + quat(4) + vel(3) + angular_vel(3)
    static constexpr int OBS_DIM = 7;     // pos(3) + quat(4)

    using StateVector = Eigen::Matrix<float, STATE_DIM, 1>;
    using ObsVector = Eigen::Matrix<float, OBS_DIM, 1>;
    using StateMatrix = Eigen::Matrix<float, STATE_DIM, STATE_DIM>;
    using ObsMatrix = Eigen::Matrix<float, OBS_DIM, OBS_DIM>;
    using ObsJacobian = Eigen::Matrix<float, OBS_DIM, STATE_DIM>;
    using KalmanGain = Eigen::Matrix<float, STATE_DIM, OBS_DIM>;

    PoseEKF() {
        // Initialize state vector
        this->x_.setZero();
        this->x_(3) = 1.0f;  // Set quaternion w component to 1

        // Initialize covariance matrix
        this->P_.setIdentity();
        this->P_ *= 0.1f;

        // Initialize process and observation noise
        set_default_noise_matrices();

        this->is_initialized_ = false;
    }

    // Set diagonal components of process noise
    void set_process_noise(const StateVector& process_noise_diag) { this->Q_ = process_noise_diag.asDiagonal(); }

    // Set diagonal components of observation noise
    void set_observation_noise(const ObsVector& obs_noise_diag) { this->R_ = obs_noise_diag.asDiagonal(); }

    /// @brief predict no input
    /// @return success or not
    bool predict(double timestamp) {
        if (!this->is_initialized_) return false;

        const auto dt = this->calc_predict_timestamp(timestamp);

        // Calculate state transition
        const StateVector x_pred = state_transition(this->x_, dt);

        // Calculate Jacobian matrix
        const StateMatrix F = compute_state_jacobian(this->x_, dt);

        // Update covariance matrix
        this->P_ = F * this->P_ * F.transpose() + this->Q_;

        // Update state vector
        this->x_ = x_pred;

        // Normalize quaternion
        normalize_quaternion();
        return true;
    }

    // Update step (6DoF pose from LiDAR odometry)
    void update(const Eigen::Isometry3f& odom) {
        const Eigen::Vector3f observed_position(odom.translation());
        const Eigen::Quaternionf observed_orientation(odom.rotation());

        if (!this->is_initialized_) {
            set_initial_pose(odom);
            return;
        }

        // Build observation vector
        ObsVector z;
        z.segment<3>(0) = observed_position;
        z(3) = observed_orientation.w();
        z(4) = observed_orientation.x();
        z(5) = observed_orientation.y();
        z(6) = observed_orientation.z();

        // Calculate predicted observation
        const ObsVector h = observation_model(this->x_);

        // Calculate observation Jacobian
        const ObsJacobian H = compute_observation_jacobian(this->x_);

        // Innovation covariance matrix
        const ObsMatrix S = H * this->P_ * H.transpose() + this->R_;

        // Calculate Kalman gain
        const KalmanGain K = this->P_ * H.transpose() * (S + ObsMatrix::Identity() * 1e-4f).inverse();

        // Innovation vector (quaternion difference requires special treatment)
        const ObsVector innovation = compute_innovation(z, h);

        // Update state vector
        this->x_ += K * innovation;

        // Update covariance matrix
        StateMatrix I = StateMatrix::Identity();
        this->P_ = (I - K * H) * this->P_;
        this->P_ = 0.5f * (this->P_ + this->P_.transpose());  // ensure symmetric

        // Normalize quaternion
        normalize_quaternion();
    }

    // Get current state
    Eigen::Vector3f get_position_state() const { return this->x_.segment<3>(0); }

    Eigen::Quaternionf get_quaternion_state() const {
        return Eigen::Quaternionf(this->x_(3), this->x_(4), this->x_(5), this->x_(6));
    }

    Eigen::Vector3f get_velocity_state() const { return this->x_.segment<3>(7); }

    Eigen::Vector3f get_angular_velocity_state() const { return this->x_.segment<3>(10); }

    const StateVector& get_state() const { return this->x_; }

    const StateMatrix& get_covariance() const { return this->P_; }

    const Eigen::Isometry3f get_pose() const {
        Eigen::Isometry3f pose;
        pose.translation() = this->get_position_state();
        pose.matrix().block<3, 3>(0, 0) = this->get_quaternion_state().matrix();
        return pose;
    }

private:
    StateVector x_;  // State vector
    StateMatrix P_;  // Covariance matrix
    StateMatrix Q_;  // Process noise covariance matrix
    ObsMatrix R_;    // Observation noise covariance matrix

    double last_predict_timestamp_ = -1.0;
    const double default_predict_dt_ = 0.1;  // 10Hz
    bool is_initialized_;

    // Initialize state vector
    void set_initial_pose(const Eigen::Isometry3f& pose) {
        this->x_.setZero();

        const Eigen::Quaternionf quat(pose.rotation());

        // Position
        this->x_.segment<3>(0) = pose.translation();

        // Quaternion (w, x, y, z)
        this->x_(3) = quat.w();
        this->x_(4) = quat.x();
        this->x_(5) = quat.y();
        this->x_(6) = quat.z();

        // Initialize velocity and angular velocity to zero
        // x_.segment<3>(7) = velocity (already zero)
        // x_.segment<3>(10) = angular_velocity (already zero)

        this->is_initialized_ = true;
    }

    double calc_predict_timestamp(double timestamp) {
        if (this->last_predict_timestamp_ < 0.0) {
            this->last_predict_timestamp_ = timestamp;
            return this->default_predict_dt_;
        };
        if (timestamp < this->last_predict_timestamp_) {
            throw std::runtime_error("Invalid timestamp.");
        }
        const auto dt = timestamp - this->last_predict_timestamp_;
        this->last_predict_timestamp_ = timestamp;
        return dt;
    }

    // Set default noise matrices
    void set_default_noise_matrices() {
        // Process noise (diagonal components)
        StateVector process_noise;
        process_noise << 0.01f, 0.01f, 0.01f,  // Position
            0.001f, 0.001f, 0.001f, 0.001f,    // Quaternion
            0.1f, 0.1f, 0.1f,                  // Velocity
            0.01f, 0.01f, 0.01f;               // Angular velocity
        set_process_noise(process_noise);

        // Observation noise (diagonal components)
        ObsVector obs_noise;
        obs_noise << 0.05f, 0.05f, 0.05f,  // Position
            0.01f, 0.01f, 0.01f, 0.01f;    // Quaternion
        set_observation_noise(obs_noise);
    }

    // State transition function
    StateVector state_transition(const StateVector& x, double dt) const {
        StateVector x_next = x;  // copy

        // current state
        const Eigen::Vector3f current_position = x.segment<3>(0);
        const Eigen::Vector3f current_velocity = x.segment<3>(7);
        const Eigen::Vector3f current_angular_vel = x.segment<3>(10);
        const Eigen::Quaternionf current_q(x(3), x(4), x(5), x(6));

        // Update position: p += v * dt
        x_next.segment<3>(0) += current_velocity * dt;

        // Update quaternion
        const float angle = current_angular_vel.norm() * dt;

        if (angle > std::numeric_limits<float>::epsilon()) {
            const Eigen::Vector3f axis = current_angular_vel.normalized();
            const Eigen::Quaternionf delta_q(Eigen::AngleAxisf(angle, axis));

            const Eigen::Quaternionf new_q = current_q * delta_q;

            x_next(3) = new_q.w();
            x_next(4) = new_q.x();
            x_next(5) = new_q.y();
            x_next(6) = new_q.z();
        } else {
            Eigen::Quaternionf delta_q;
            delta_q.w() = 1.0f;
            delta_q.x() = 0.5f * dt * current_angular_vel.x();
            delta_q.y() = 0.5f * dt * current_angular_vel.y();
            delta_q.z() = 0.5f * dt * current_angular_vel.z();
            delta_q.normalize();

            const Eigen::Quaternionf new_q = current_q * delta_q;
            x_next(3) = new_q.w();
            x_next(4) = new_q.x();
            x_next(5) = new_q.y();
            x_next(6) = new_q.z();
        }

        // Velocity and angular velocity use constant model (no input)
        // x_next.segment<3>(7) = x.segment<3>(7);
        // x_next.segment<3>(10) = x.segment<3>(10);

        return x_next;
    }

    // Calculate Jacobian matrix of state transition
    StateMatrix compute_state_jacobian(const StateVector& x, double dt) const {
        StateMatrix F = StateMatrix::Identity();

        // ∂p/∂v = I * dt
        F.block<3, 3>(0, 7) = Eigen::Matrix3f::Identity() * dt;

        // Quaternion derivative with respect to angular velocity: ∂q/∂ω
        const Eigen::Vector3f angular_vel = x.segment<3>(10);
        const Eigen::Quaternionf q(x(3), x(4), x(5), x(6));

        // Quaternion derivative matrix: q̇ = 0.5 * Ω(ω) * q
        // Where Ω(ω) is the quaternion multiplication matrix for angular velocity
        Eigen::Matrix<float, 4, 3> Q_omega;
        Q_omega << -q.x(), -q.y(), -q.z(),  // ∂qw/∂ω
            q.w(), -q.z(), q.y(),           // ∂qx/∂ω
            q.z(), q.w(), -q.x(),           // ∂qy/∂ω
            -q.y(), q.x(), q.w();           // ∂qz/∂ω

        F.block<4, 3>(3, 10) = 0.5f * dt * Q_omega;

        // Quaternion derivative with respect to current quaternion: ∂q/∂q
        if (angular_vel.norm() > std::numeric_limits<float>::epsilon()) {
            const float angle = angular_vel.norm() * dt;
            const Eigen::Vector3f axis = angular_vel.normalized();
            const float half_angle = angle * 0.5f;
            const float cos_half = std::cos(half_angle);
            const float sin_half = std::sin(half_angle);

            Eigen::Vector4f delta_q;
            delta_q << cos_half, sin_half * axis.x(), sin_half * axis.y(), sin_half * axis.z();

            Eigen::Matrix4f Q_quat = Eigen::Matrix4f::Identity();
            Q_quat << delta_q(0), -delta_q(1), -delta_q(2), -delta_q(3),  //
                delta_q(1), delta_q(0), -delta_q(3), delta_q(2),          //
                delta_q(2), delta_q(3), delta_q(0), -delta_q(1),          //
                delta_q(3), -delta_q(2), delta_q(1), delta_q(0);

            F.block<4, 4>(3, 3) = Q_quat;
        } else {
            // If angular vel is too small, first approximation.
            /* q(k+1) ≒ q(k) + 0.5 * dt * Ω(ω) * q(k) */
            /* ∂q(k+1)/∂q(k) = I + 0.5 * dt * Ω(ω) */
            Eigen::Matrix4f Q_quat = Eigen::Matrix4f::Identity();
            const float wx_dt = angular_vel(0) * dt * 0.5f;
            const float wy_dt = angular_vel(1) * dt * 0.5f;
            const float wz_dt = angular_vel(2) * dt * 0.5f;

            Q_quat(0, 1) = -wx_dt;  // ∂qw/∂qx
            Q_quat(0, 2) = -wy_dt;  // ∂qw/∂qy
            Q_quat(0, 3) = -wz_dt;  // ∂qw/∂qz

            Q_quat(1, 0) = wx_dt;   // ∂qx/∂qw
            Q_quat(1, 2) = wz_dt;   // ∂qx/∂qy
            Q_quat(1, 3) = -wy_dt;  // ∂qx/∂qz

            Q_quat(2, 0) = wy_dt;   // ∂qy/∂qw
            Q_quat(2, 1) = -wz_dt;  // ∂qy/∂qx
            Q_quat(2, 3) = wx_dt;   // ∂qy/∂qz

            Q_quat(3, 0) = wx_dt;   // ∂qz/∂qw
            Q_quat(3, 1) = wy_dt;   // ∂qz/∂qx
            Q_quat(3, 2) = -wz_dt;  // ∂qz/∂qy

            F.block<4, 4>(3, 3) = Q_quat;
        }

        return F;
    }

    // Observation model
    ObsVector observation_model(const StateVector& x) const {
        ObsVector h = ObsVector::Zero();
        h.segment<3>(0) = x.segment<3>(0);  // Position
        h.segment<4>(3) = x.segment<4>(3);  // Quaternion
        return h;
    }

    // Calculate observation Jacobian
    ObsJacobian compute_observation_jacobian(const StateVector& x) const {
        ObsJacobian H = ObsJacobian::Zero();

        // Position observation
        H.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity();

        // Quaternion observation
        H.block<4, 4>(3, 3) = Eigen::Matrix4f::Identity();

        return H;
    }

    // Calculate innovation vector (considering quaternion difference)
    ObsVector compute_innovation(const ObsVector& z, const ObsVector& h) const {
        ObsVector innovation;

        // Position difference
        innovation.segment<3>(0) = z.segment<3>(0) - h.segment<3>(0);

        // Quaternion difference (special treatment)
        const Eigen::Quaternionf z_q(z(3), z(4), z(5), z(6));
        const Eigen::Quaternionf h_q(h(3), h(4), h(5), h(6));

        // Express quaternion difference as rotation vector
        Eigen::Quaternionf error_q = z_q * h_q.inverse();
        error_q.normalize();

        // Method 1: Use rotation vector representation (recommended)
        // Small angle approximation: δθ ≈ 2 * [qx, qy, qz] (when qw ≈ 1)
        if (std::abs(1.0f - error_q.w()) < 0.01f) {
            // Small angle approximation is valid
            innovation.segment<4>(3) << 0.0f,  // qw component is 0 (small angle)
                2.0f * error_q.x(), 2.0f * error_q.y(), 2.0f * error_q.z();
        } else {
            // Large angle case: full rotation vector conversion
            const float angle = 2.0f * std::atan2(error_q.vec().norm(), error_q.w());
            // const float angle = 2.0f *  std::acos(std::abs(error_q.w()));
            const float sin_half_angle = std::sqrt(1.0f - error_q.w() * error_q.w());

            if (sin_half_angle > std::numeric_limits<float>::epsilon()) {
                const float scale = angle / sin_half_angle;
                innovation.segment<4>(3) << 0.0f,  // qw component is 0
                    scale * error_q.x(), scale * error_q.y(), scale * error_q.z();
            } else {
                // Angle is nearly zero
                innovation.segment<4>(3) << 0.0f, 0.0f, 0.0f, 0.0f;
            }
        }
        return innovation;
    }

    // Normalize quaternion
    void normalize_quaternion() {
        Eigen::Quaternionf quat = get_quaternion_state();
        quat.normalize();
        x_(3) = quat.w();
        x_(4) = quat.x();
        x_(5) = quat.y();
        x_(6) = quat.z();
    }
};

}  // namespace ekf
}  // namespace sycl_points