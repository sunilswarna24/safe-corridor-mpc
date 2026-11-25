// Closed-loop unicycle + corridor MPC + gate on a straight aisle (no map IO).
#include "safe_corridor/gate.hpp"
#include "safe_corridor/mpc.hpp"
#include "safe_corridor/plant.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace safe_corridor;

static DiskObstacle mover_at(double t) {
  const double speed = 0.7;
  const double phase = -2.5;
  const double s = std::clamp((t - phase) * speed, 0.0, 6.0);
  return DiskObstacle{-3.0 + s, 2.0, 0.45};
}

int main() {
  MpcConfig mpc_cfg;
  mpc_cfg.N = 40;
  mpc_cfg.use_corridors = true;
  UnicycleMpc mpc(mpc_cfg);
  SupervisoryGate gate;

  const double dt = mpc_cfg.dt;
  const double v_cruise = 0.55;
  const double goal_y = 6.0;
  const int max_ticks = 500;

  State z{0.0, 0.0, 1.5707963267948966, 0.0};
  int hits = 0, sealed = 0, ok_n = 0, ticks = 0;
  double err2 = 0.0;

  for (int k = 0; k < max_ticks; ++k) {
    const double t = k * dt;
    if (z[1] >= goal_y - 0.05 && z[3] < 0.05) break;

    std::vector<State> ref(mpc_cfg.N);
    for (int j = 0; j < mpc_cfg.N; ++j) {
      const double yy = std::min(z[1] + v_cruise * (j + 1) * dt, goal_y);
      const double vv = (yy >= goal_y - 1e-6) ? 0.0 : v_cruise;
      ref[j] = State{0.0, yy, 1.5707963267948966, vv};
    }

    std::vector<Box> boxes(mpc_cfg.N);
    for (int j = 0; j < mpc_cfg.N; ++j) {
      const DiskObstacle m = mover_at(t + (j + 1) * dt);
      const double y = ref[j][1];
      const double dx = m.x - 0.0;
      const double dy = m.y - y;
      if (dx * dx + dy * dy <= (m.radius + 0.5) * (m.radius + 0.5)) {
        boxes[j] = Box{-0.02, 0.02, y - 0.02, y + 0.02};
      } else {
        boxes[j] = Box{-1.0, 1.0, y - 1.0, y + 1.0};
      }
    }

    const DiskObstacle now = mover_at(t);
    auto [allow, reason] = gate.pre_check(z, boxes, {now});
    if (std::strcmp(reason, "sealed_corridor") == 0) ++sealed;

    SolveInfo info;
    Input u_mpc = mpc.solve(z, ref, &boxes, &info);
    if (info.ok) ++ok_n;
    Input u = gate.post_command(u_mpc, z, info.ok, allow, reason);
    z = step(z, u, dt, 1.0, mpc_cfg.a_max, mpc_cfg.w_max);

    err2 += z[0] * z[0];
    if (std::hypot(z[0] - now.x, z[1] - now.y) < now.radius + 0.30) ++hits;
    ++ticks;
  }

  std::printf("cpp demo_line\n");
  std::printf("  ticks=%d  hits=%d  sealed=%d  wait=%d  fb=%d\n", ticks, hits, sealed,
              gate.wait_count(), gate.fallback_count());
  std::printf("  rms_x=%.4f  ok_frac=%.3f\n", std::sqrt(err2 / std::max(ticks, 1)),
              static_cast<double>(ok_n) / std::max(ticks, 1));
  std::printf("  final y=%.3f  v=%.3f\n", z[1], z[3]);
  return hits > 0 ? 1 : 0;
}
