// 20 Hz unicycle corridor MPC node.
// Subscribes: odom (nav_msgs/Odometry), path (nav_msgs/Path)
// Publishes: cmd_vel (geometry_msgs/Twist) as [a along body-x approximation → v cmd via integrate]
//
// cmd_vel uses linear.x = commanded speed after integrating a, angular.z = omega.
// Gate may hard-brake. Corridors are axis-aligned boxes of half-width `corridor_half`
// along the path; optional /movers as PoseArray disks (radius param).

#include "safe_corridor/gate.hpp"
#include "safe_corridor/mpc.hpp"
#include "safe_corridor/plant.hpp"

#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <vector>

using safe_corridor::Box;
using safe_corridor::DiskObstacle;
using safe_corridor::GateConfig;
using safe_corridor::Input;
using safe_corridor::MpcConfig;
using safe_corridor::SolveInfo;
using safe_corridor::State;
using safe_corridor::SupervisoryGate;
using safe_corridor::UnicycleMpc;
using safe_corridor::wrap_angle;

class MpcNode : public rclcpp::Node {
 public:
  MpcNode() : Node("safe_corridor_mpc") {
    corridor_half_ = declare_parameter<double>("corridor_half", 1.0);
    mover_radius_ = declare_parameter<double>("mover_radius", 0.45);
    use_gate_ = declare_parameter<bool>("use_gate", true);

    MpcConfig cfg;
    cfg.use_corridors = true;
    mpc_ = std::make_unique<UnicycleMpc>(cfg);
    GateConfig gcfg;
    gate_ = std::make_unique<SupervisoryGate>(gcfg);
    dt_ = cfg.dt;
    N_ = cfg.N;
    a_max_ = cfg.a_max;
    w_max_ = cfg.w_max;

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        "odom", 10, [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
          std::lock_guard<std::mutex> lock(mu_);
          const auto& p = msg->pose.pose.position;
          const auto& q = msg->pose.pose.orientation;
          const double yaw = std::atan2(
              2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
          const double vx = msg->twist.twist.linear.x;
          const double vy = msg->twist.twist.linear.y;
          z_ = State{p.x, p.y, yaw, std::hypot(vx, vy)};
          have_odom_ = true;
        });

    path_sub_ = create_subscription<nav_msgs::msg::Path>(
        "path", 10, [this](const nav_msgs::msg::Path::SharedPtr msg) {
          std::lock_guard<std::mutex> lock(mu_);
          path_.clear();
          for (const auto& ps : msg->poses) {
            path_.push_back({ps.pose.position.x, ps.pose.position.y});
          }
        });

    movers_sub_ = create_subscription<geometry_msgs::msg::PoseArray>(
        "movers", 10, [this](const geometry_msgs::msg::PoseArray::SharedPtr msg) {
          std::lock_guard<std::mutex> lock(mu_);
          movers_.clear();
          for (const auto& p : msg->poses) {
            movers_.push_back(DiskObstacle{p.position.x, p.position.y, mover_radius_});
          }
        });

    cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 10);
    timer_ = create_wall_timer(std::chrono::milliseconds(50),
                               std::bind(&MpcNode::on_timer, this));
    RCLCPP_INFO(get_logger(), "safe_corridor_mpc @ 20 Hz");
  }

 private:
  void on_timer() {
    State z;
    std::vector<std::array<double, 2>> path;
    std::vector<DiskObstacle> movers;
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (!have_odom_ || path_.size() < 2) return;
      z = z_;
      path = path_;
      movers = movers_;
    }

    // Nearest path index
    int nearest = 0;
    double best = 1e9;
    for (size_t i = 0; i < path.size(); ++i) {
      const double d = std::hypot(path[i][0] - z[0], path[i][1] - z[1]);
      if (d < best) {
        best = d;
        nearest = static_cast<int>(i);
      }
    }

    std::vector<State> ref(N_);
    std::vector<Box> boxes(N_);
    for (int j = 0; j < N_; ++j) {
      const int idx = std::min(nearest + j + 1, static_cast<int>(path.size()) - 1);
      const int idx1 = std::min(idx + 1, static_cast<int>(path.size()) - 1);
      const double x = path[idx][0];
      const double y = path[idx][1];
      double th = z[2];
      if (idx1 != idx) th = std::atan2(path[idx1][1] - y, path[idx1][0] - x);
      double v = 0.55;
      if (idx >= static_cast<int>(path.size()) - 1) v = 0.0;
      ref[j] = State{x, y, th, v};

      bool sealed = false;
      for (const auto& m : movers) {
        const double dx = m.x - x;
        const double dy = m.y - y;
        if (dx * dx + dy * dy <= (m.radius + 0.5) * (m.radius + 0.5)) {
          sealed = true;
          break;
        }
      }
      if (sealed) {
        boxes[j] = Box{x - 0.02, x + 0.02, y - 0.02, y + 0.02};
      } else {
        boxes[j] = Box{x - corridor_half_, x + corridor_half_, y - corridor_half_,
                       y + corridor_half_};
      }
    }

    bool allow = true;
    const char* reason = "ok";
    if (use_gate_) {
      auto pr = gate_->pre_check(z, boxes, movers);
      allow = pr.first;
      reason = pr.second;
    }

    SolveInfo info;
    Input u_mpc = mpc_->solve(z, ref, &boxes, &info);
    Input u = u_mpc;
    if (use_gate_) u = gate_->post_command(u_mpc, z, info.ok, allow, reason);
    else if (!info.ok) u = Input{0.0, 0.0};

    // Integrate accel → speed command for diff-drive style Twist
    speed_cmd_ = std::clamp(speed_cmd_ + dt_ * u[0], 0.0, 1.0);
    if (!allow || !info.ok) speed_cmd_ = std::max(0.0, speed_cmd_ + dt_ * u[0]);

    geometry_msgs::msg::Twist cmd;
    cmd.linear.x = speed_cmd_;
    cmd.angular.z = u[1];
    cmd_pub_->publish(cmd);
  }

  double corridor_half_ = 1.0;
  double mover_radius_ = 0.45;
  bool use_gate_ = true;
  double dt_ = 0.05;
  int N_ = 40;
  double a_max_ = 1.5;
  double w_max_ = 1.2;
  double speed_cmd_ = 0.0;

  std::unique_ptr<UnicycleMpc> mpc_;
  std::unique_ptr<SupervisoryGate> gate_;

  std::mutex mu_;
  State z_{};
  bool have_odom_ = false;
  std::vector<std::array<double, 2>> path_;
  std::vector<DiskObstacle> movers_;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr movers_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MpcNode>());
  rclcpp::shutdown();
  return 0;
}
