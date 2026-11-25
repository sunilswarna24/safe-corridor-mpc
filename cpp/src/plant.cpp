#include "safe_corridor/plant.hpp"

#include <algorithm>
#include <cmath>

namespace safe_corridor {

double wrap_angle(double a) {
  const double two_pi = 2.0 * 3.14159265358979323846;
  a = std::fmod(a + 3.14159265358979323846, two_pi);
  if (a < 0.0) a += two_pi;
  return a - 3.14159265358979323846;
}

State f_discrete(const State& z, const Input& u, double dt) {
  const double x = z[0], y = z[1], th = z[2], v = z[3];
  const double a = u[0], w = u[1];
  return State{
      x + dt * v * std::cos(th),
      y + dt * v * std::sin(th),
      th + dt * w,
      v + dt * a,
  };
}

void linearize(const State& z, const Input& u, double dt, double A[kNx][kNx],
               double B[kNx][kNu], double c[kNx]) {
  const double th = z[2], v = z[3];
  const double ct = std::cos(th), st = std::sin(th);

  for (int i = 0; i < kNx; ++i) {
    for (int j = 0; j < kNx; ++j) A[i][j] = (i == j) ? 1.0 : 0.0;
    for (int j = 0; j < kNu; ++j) B[i][j] = 0.0;
  }
  A[0][2] = -dt * v * st;
  A[0][3] = dt * ct;
  A[1][2] = dt * v * ct;
  A[1][3] = dt * st;
  B[2][1] = dt;
  B[3][0] = dt;

  const State zp = f_discrete(z, u, dt);
  for (int i = 0; i < kNx; ++i) {
    double Az = 0.0, Bu = 0.0;
    for (int j = 0; j < kNx; ++j) Az += A[i][j] * z[j];
    for (int j = 0; j < kNu; ++j) Bu += B[i][j] * u[j];
    c[i] = zp[i] - Az - Bu;
  }
}

State step(const State& z, const Input& u_in, double dt, double v_max, double a_max,
           double w_max) {
  Input u = u_in;
  u[0] = std::clamp(u[0], -a_max, a_max);
  u[1] = std::clamp(u[1], -w_max, w_max);
  State zn = f_discrete(z, u, dt);
  zn[2] = wrap_angle(zn[2]);
  zn[3] = std::clamp(zn[3], 0.0, v_max);
  return zn;
}

}  // namespace safe_corridor
