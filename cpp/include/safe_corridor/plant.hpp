#pragma once

#include "safe_corridor/types.hpp"

namespace safe_corridor {

double wrap_angle(double a);

State f_discrete(const State& z, const Input& u, double dt);

// z+ ≈ A z + B u + c at operating point (z,u)
void linearize(const State& z, const Input& u, double dt,
               double A[kNx][kNx], double B[kNx][kNu], double c[kNx]);

State step(const State& z, const Input& u, double dt, double v_max, double a_max,
           double w_max);

}  // namespace safe_corridor
