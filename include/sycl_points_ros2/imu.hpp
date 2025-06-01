#pragma once

#include <Eigen/Dense>
#include <sensor_msgs/msg/imu.hpp>
#include <tuple>

namespace sycl_points {
namespace imu {
using LinearAccceleration = Eigen::Vector3f;
using AngularVelocity = Eigen::Vector3f;


struct IMU {
    double timestamp = 0.0;
    LinearAccceleration linear_acceleration = LinearAccceleration::Zero();
    AngularVelocity angular_velocity = AngularVelocity::Zero();
};

inline IMU fromROS2msg(const sensor_msgs::msg::Imu& msg) {
    const double timestamp =
        static_cast<double>(msg.header.stamp.sec) + static_cast<double>(msg.header.stamp.nanosec) * 1e-9;
    const LinearAccceleration linear_acceleration(msg.linear_acceleration.x, msg.linear_acceleration.y,
                                                  msg.linear_acceleration.z);
    const AngularVelocity angular_velocity(msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z);
    return {timestamp, linear_acceleration, angular_velocity};
}

}  // namespace imu
}  // namespace sycl_points