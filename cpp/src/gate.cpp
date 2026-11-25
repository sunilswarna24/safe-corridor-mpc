#include "safe_corridor/gate.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace safe_corridor {

SupervisoryGate::SupervisoryGate(const GateConfig& cfg) : cfg_(cfg) {}

std::pair<bool, const char*> SupervisoryGate::pre_check(
    const State& z, const std::vector<Box>& corridors,
    const std::vector<DiskObstacle>& movers) const {
  bool sealed = false;
  const int look = std::min(20, static_cast<int>(corridors.size()));
  for (int i = 0; i < look; ++i) {
    const auto& b = corridors[i];
    const double hx = 0.5 * (b[1] - b[0]);
    const double hy = 0.5 * (b[3] - b[2]);
    if (std::min(hx, hy) < cfg_.min_corridor_half) {
      sealed = true;
      break;
    }
  }

  const double fx = std::cos(z[2]);
  const double fy = std::sin(z[2]);
  bool any_close_ahead = false;
  double min_dist = 1e9;
  double r_max = 0.0;
  for (const auto& m : movers) {
    r_max = std::max(r_max, m.radius);
    const double dx = m.x - z[0];
    const double dy = m.y - z[1];
    const double dist = std::hypot(dx, dy);
    min_dist = std::min(min_dist, dist);
    const double stop_d = m.radius + cfg_.robot_radius + cfg_.stop_clear;
    const double fwd = dx * fx + dy * fy;
    if (fwd > 0.5 * std::max(dist, 1e-6) && dist <= stop_d) any_close_ahead = true;
  }

  const double resume_d = r_max + cfg_.robot_radius + cfg_.resume_clear;
  if (waiting_) {
    if (min_dist >= resume_d || !any_close_ahead) waiting_ = false;
    else return {false, "mover_close"};
  } else if (any_close_ahead) {
    waiting_ = true;
    return {false, "mover_close"};
  }

  if (sealed) return {false, "sealed_corridor"};
  return {true, "ok"};
}

Input SupervisoryGate::brake(const State& z) const {
  const double a = std::clamp(-z[3] / cfg_.dt, -cfg_.a_max, cfg_.a_max);
  return Input{a, 0.0};
}

Input SupervisoryGate::post_command(const Input& u_mpc, const State& z, bool qp_ok,
                                    bool allow, const char* reason) {
  Input u;
  if (!allow || !qp_ok) {
    u = brake(z);
    ++fallback_count_;
    if (!allow && reason && std::strcmp(reason, "mover_close") == 0) ++wait_count_;
  } else {
    u = u_mpc;
  }
  Input u_cmd;
  for (int i = 0; i < kNu; ++i) {
    const double du = std::clamp(u[i] - u_prev_[i], -cfg_.slew_max, cfg_.slew_max);
    u_cmd[i] = u_prev_[i] + du;
  }
  u_prev_ = u_cmd;
  return u_cmd;
}

}  // namespace safe_corridor
