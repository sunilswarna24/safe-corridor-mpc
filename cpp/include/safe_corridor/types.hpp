#pragma once

#include <array>
#include <vector>

namespace safe_corridor {

constexpr int kNx = 4;  // x, y, theta, v
constexpr int kNu = 2;  // a, omega

using State = std::array<double, kNx>;
using Input = std::array<double, kNu>;
using Box = std::array<double, 4>;  // xmin, xmax, ymin, ymax

struct MpcConfig {
  double dt = 0.05;
  int N = 40;
  double q_pos = 20.0;
  double q_theta = 8.0;
  double q_speed = 4.0;
  double r_u = 0.1;
  double a_max = 1.5;
  double w_max = 1.2;
  bool use_corridors = true;
};

struct SolveInfo {
  bool ok = false;
  int iter = 0;
  double solve_time_s = 0.0;
  std::vector<double> u_seq;
};

struct GateConfig {
  double slew_max = 3.0;
  double min_corridor_half = 0.06;
  double stop_clear = 1.10;
  double resume_clear = 1.80;
  double a_max = 1.5;
  double dt = 0.05;
  double robot_radius = 0.30;
};

}  // namespace safe_corridor
