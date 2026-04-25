#include "rclcpp/rclcpp.hpp"
#include <string>
#include <cmath>
#include <algorithm>

#include "sensor_msgs/msg/laser_scan.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"

class WallFollow : public rclcpp::Node {

public:
    WallFollow() : Node("wall_follow_node")
    {
        scan_sub = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10,
            std::bind(&WallFollow::scan_callback, this, std::placeholders::_1));

        drive_pub = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
            "/drive", 10);
    }

private:
    // ===== PID =====
    double kp = 4.0;
    double kd = 2.2;
    double ki = 0.3;

    double prev_error = 0.0;
    double integral = 0.0;

    double prev_time = 0.0;
    double time_gap = 0.0;

    // ===== Wall follow =====
    double dist_wall = 0.5;
    double L = 1.0;
    double alpha = 0.0;

    // ===== Corner handling =====
    double CORNER_DIFF_THRESH = -0.1;
    double FRONT_NEED_FOR_TURN = 1.2;
    double TURN_STEER = 0.6;

    // ===== ROS =====
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub;

    sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg;

    int scan_count = 0;

    // ===== Get range =====
    double get_range(const float* ranges, double angle)
    {
        int index = (int)((angle - scan_msg->angle_min) / scan_msg->angle_increment);
        int size = scan_msg->ranges.size();

        if (index < 0 || index >= size)
            return scan_msg->range_max;

        float r = ranges[index];
        if (std::isnan(r) || std::isinf(r))
            return scan_msg->range_max;

        return (double)r;
    }

    // ===== Compute wall error =====
    double get_error(const float* ranges, double desired_dist)
    {
        double theta = M_PI / 4.0; // 45°

        double a = get_range(ranges, M_PI_2 - theta); // 45°
        double b = get_range(ranges, M_PI_2);         // 90°

        alpha = std::atan2(a * std::cos(theta) - b, a * std::sin(theta));
        alpha = std::clamp(alpha, -M_PI/6, M_PI/3);

        double current_dist = b * std::cos(alpha);
        double future_dist  = current_dist + L * std::sin(alpha);

        return desired_dist - future_dist;
    }

    // ===== PID control =====
    void pid_control(double error, double velocity)
    {
        double angle = 0.0;

        if (time_gap > 0) {
            integral += error * time_gap;
            double derivative = (error - prev_error) / time_gap;
            angle = kp * error + ki * integral + kd * derivative;
        }

        prev_error = error;

        ackermann_msgs::msg::AckermannDriveStamped msg;
        msg.header.stamp = this->now();
        msg.header.frame_id = "base_link";

        msg.drive.speed = velocity;
        msg.drive.steering_angle = -angle;

        drive_pub->publish(msg);
    }

    // ===== Direct steering (for corners) =====
    void publish_turn(double steer, double velocity)
    {
        ackermann_msgs::msg::AckermannDriveStamped msg;
        msg.header.stamp = this->now();
        msg.header.frame_id = "base_link";

        msg.drive.speed = velocity;
        msg.drive.steering_angle = steer;

        drive_pub->publish(msg);
    }

    // ===== Main callback =====
    void scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr msg)
    {
        scan_msg = msg;
        const float* ranges = msg->ranges.data();

        // Key beams
        double dist_left90 = get_range(ranges,  M_PI_2);
        double dist_left45 = get_range(ranges,  M_PI/4);
        double dist_front  = get_range(ranges,  0.0);

        // ===== Corner detection =====
        double diff = std::abs(dist_left90 - dist_left45);
        bool is_corner = (diff > CORNER_DIFF_THRESH) && (dist_front < FRONT_NEED_FOR_TURN);

        // ===== Velocity =====
        double velocity = (std::abs(alpha) < M_PI/18) ? 0.5 : 0.5;

        // ===== Behavior =====
        if (is_corner) {
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
                "Corner detected → turning RIGHT");

            publish_turn(-TURN_STEER, 0.5);
        }
        else {
            double error = get_error(ranges, dist_wall);
            pid_control(error, velocity);
        }

        // ===== Time update =====
        double now = this->now().seconds();
        time_gap = (prev_time > 0) ? (now - prev_time) : 0.0;
        prev_time = now;

        scan_count++;
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<WallFollow>());
    rclcpp::shutdown();
    return 0;
}
