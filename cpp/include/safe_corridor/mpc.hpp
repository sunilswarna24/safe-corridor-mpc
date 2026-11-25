#pragma once

#include <vector>

#include "safe_corridor/types.hpp"

namespace safe_corridor {

class UnicycleMpc {
 public:
  explicit UnicycleMpc(const MpcConfig& cfg);

  // ref_z: N states; corridors: N boxes if use_corridors
  Input solve(const State& z0, const std::vector<State>& ref_z,
              const std::vector<Box>* corridors, SolveInfo* info);

 private:
  MpcConfig cfg_;
  std::vector<double> u_warm_;
};

}  // namespace safe_corridor
