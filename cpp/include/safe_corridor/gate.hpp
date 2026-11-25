#pragma once

#include <utility>
#include <vector>

#include "safe_corridor/types.hpp"

namespace safe_corridor {

struct DiskObstacle {
  double x = 0.0;
  double y = 0.0;
  double radius = 0.4;
};

class SupervisoryGate {
 public:
  explicit SupervisoryGate(const GateConfig& cfg = {});

  // Returns (allow_mpc, reason) reason: "ok" | "mover_close" | "sealed_corridor"
  std::pair<bool, const char*> pre_check(const State& z,
                                         const std::vector<Box>& corridors,
                                         const std::vector<DiskObstacle>& movers) const;

  Input brake(const State& z) const;
  Input post_command(const Input& u_mpc, const State& z, bool qp_ok, bool allow,
                     const char* reason);

  int fallback_count() const { return fallback_count_; }
  int wait_count() const { return wait_count_; }
  bool waiting() const { return waiting_; }

 private:
  GateConfig cfg_;
  Input u_prev_{{0.0, 0.0}};
  mutable bool waiting_ = false;
  int fallback_count_ = 0;
  int wait_count_ = 0;
};

}  // namespace safe_corridor
