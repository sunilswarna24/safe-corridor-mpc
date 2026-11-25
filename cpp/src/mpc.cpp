#include "safe_corridor/mpc.hpp"

#include "safe_corridor/plant.hpp"

#include <Eigen/Dense>
#include <osqp.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace safe_corridor {
namespace {

using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;

void ref_inputs(const std::vector<State>& ref, double dt, std::vector<Input>* U) {
  const int N = static_cast<int>(ref.size());
  U->assign(N, Input{0.0, 0.0});
  for (int k = 0; k < N - 1; ++k) {
    (*U)[k][0] = (ref[k + 1][3] - ref[k][3]) / dt;
    (*U)[k][1] = wrap_angle(ref[k + 1][2] - ref[k][2]) / dt;
  }
  if (N >= 2) (*U)[N - 1] = (*U)[N - 2];
}

std::vector<State> align_heading(double z0_th, const std::vector<State>& ref_in) {
  std::vector<State> ref = ref_in;
  ref[0][2] = z0_th + wrap_angle(ref[0][2] - z0_th);
  for (size_t i = 1; i < ref.size(); ++i) {
    ref[i][2] = ref[i - 1][2] + wrap_angle(ref_in[i][2] - ref_in[i - 1][2]);
  }
  return ref;
}

void dense_to_csc(const Mat& M, std::vector<c_float>* data, std::vector<c_int>* indices,
                  std::vector<c_int>* indptr) {
  const int rows = static_cast<int>(M.rows());
  const int cols = static_cast<int>(M.cols());
  data->clear();
  indices->clear();
  indptr->assign(1, 0);
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      const double v = M(i, j);
      if (std::abs(v) > 1e-14) {
        data->push_back(static_cast<c_float>(v));
        indices->push_back(static_cast<c_int>(i));
      }
    }
    indptr->push_back(static_cast<c_int>(data->size()));
  }
}

}  // namespace

UnicycleMpc::UnicycleMpc(const MpcConfig& cfg) : cfg_(cfg) {
  u_warm_.assign(static_cast<size_t>(cfg_.N * kNu), 0.0);
}

Input UnicycleMpc::solve(const State& z0, const std::vector<State>& ref_in,
                         const std::vector<Box>* corridors, SolveInfo* info) {
  SolveInfo local;
  SolveInfo& out = info ? *info : local;
  out = SolveInfo{};
  out.u_seq.assign(static_cast<size_t>(cfg_.N * kNu), 0.0);

  const int N = cfg_.N;
  const int nx = kNx;
  const int nu = kNu;
  if (static_cast<int>(ref_in.size()) != N) return Input{0.0, 0.0};
  if (cfg_.use_corridors &&
      (!corridors || static_cast<int>(corridors->size()) != N)) {
    return Input{0.0, 0.0};
  }

  auto ref = align_heading(z0[2], ref_in);
  std::vector<Input> u_ref;
  ref_inputs(ref, cfg_.dt, &u_ref);

  std::vector<Mat> A_list(N, Mat::Zero(nx, nx));
  std::vector<Mat> B_list(N, Mat::Zero(nx, nu));
  std::vector<Vec> c_list(N, Vec::Zero(nx));
  for (int k = 0; k < N; ++k) {
    double A[kNx][kNx], B[kNx][kNu], c[kNx];
    linearize(ref[k], u_ref[k], cfg_.dt, A, B, c);
    for (int i = 0; i < nx; ++i) {
      c_list[k](i) = c[i];
      for (int j = 0; j < nx; ++j) A_list[k](i, j) = A[i][j];
      for (int j = 0; j < nu; ++j) B_list[k](i, j) = B[i][j];
    }
  }

  Mat S = Mat::Zero(nx * N, nx);
  Mat G = Mat::Zero(nx * N, nu * N);
  Vec d = Vec::Zero(nx * N);
  for (int k = 0; k < N; ++k) {
    if (k == 0) {
      S.block(0, 0, nx, nx) = A_list[0];
      d.segment(0, nx) = c_list[0];
    } else {
      S.block(k * nx, 0, nx, nx) = A_list[k] * S.block((k - 1) * nx, 0, nx, nx);
      d.segment(k * nx, nx) = A_list[k] * d.segment((k - 1) * nx, nx) + c_list[k];
    }
    for (int j = 0; j <= k; ++j) {
      Mat P = Mat::Identity(nx, nx);
      for (int t = j + 1; t <= k; ++t) P = A_list[t] * P;
      G.block(k * nx, j * nu, nx, nu) = P * B_list[j];
    }
  }

  Vec q_diag(nx);
  q_diag << cfg_.q_pos, cfg_.q_pos, cfg_.q_theta, cfg_.q_speed;
  Mat Qb = Mat::Zero(nx * N, nx * N);
  Mat Rb = Mat::Zero(nu * N, nu * N);
  for (int k = 0; k < N; ++k) {
    Qb.block(k * nx, k * nx, nx, nx) = q_diag.asDiagonal();
    Rb.block(k * nu, k * nu, nu, nu) =
        cfg_.r_u * Mat::Identity(nu, nu);
  }

  Vec z0v(nx);
  for (int i = 0; i < nx; ++i) z0v(i) = z0[i];
  Vec r(nx * N);
  for (int k = 0; k < N; ++k)
    for (int i = 0; i < nx; ++i) r(k * nx + i) = ref[k][i];

  const Vec offset = S * z0v + d - r;
  Mat H = G.transpose() * Qb * G + Rb;
  H = 0.5 * (H + H.transpose());
  Mat Pmat = 2.0 * H;
  Vec qvec = 2.0 * (G.transpose() * Qb * offset);

  const int nU = nu * N;
  Vec l_u(nU), u_u(nU);
  for (int k = 0; k < N; ++k) {
    l_u(2 * k) = -cfg_.a_max;
    u_u(2 * k) = cfg_.a_max;
    l_u(2 * k + 1) = -cfg_.w_max;
    u_u(2 * k + 1) = cfg_.w_max;
  }

  Mat Acon;
  Vec l, u;
  if (cfg_.use_corridors) {
    Mat Sel = Mat::Zero(2 * N, nx * N);
    for (int k = 0; k < N; ++k) {
      Sel(2 * k, nx * k) = 1.0;
      Sel(2 * k + 1, nx * k + 1) = 1.0;
    }
    const Mat SG = Sel * G;
    const Vec base = Sel * (S * z0v + d);
    Vec l_pos(2 * N), u_pos(2 * N);
    for (int k = 0; k < N; ++k) {
      const Box& b = (*corridors)[k];
      l_pos(2 * k) = b[0] - base(2 * k);
      u_pos(2 * k) = b[1] - base(2 * k);
      l_pos(2 * k + 1) = b[2] - base(2 * k + 1);
      u_pos(2 * k + 1) = b[3] - base(2 * k + 1);
    }
    Acon = Mat::Zero(nU + 2 * N, nU);
    Acon.topRows(nU) = Mat::Identity(nU, nU);
    Acon.bottomRows(2 * N) = SG;
    l = Vec(nU + 2 * N);
    u = Vec(nU + 2 * N);
    l << l_u, l_pos;
    u << u_u, u_pos;
  } else {
    Acon = Mat::Identity(nU, nU);
    l = l_u;
    u = u_u;
  }

  // OSQP wants upper-triangular P in CSC
  Mat P_triu = Mat::Zero(nU, nU);
  for (int i = 0; i < nU; ++i)
    for (int j = i; j < nU; ++j) P_triu(i, j) = Pmat(i, j);

  std::vector<c_float> P_x, A_x, q_c(nU), l_c(l.size()), u_c(u.size());
  std::vector<c_int> P_i, P_p, A_i, A_p;
  dense_to_csc(P_triu, &P_x, &P_i, &P_p);
  dense_to_csc(Acon, &A_x, &A_i, &A_p);
  for (int i = 0; i < nU; ++i) q_c[i] = static_cast<c_float>(qvec(i));
  for (Eigen::Index i = 0; i < l.size(); ++i) {
    l_c[i] = static_cast<c_float>(l(i));
    u_c[i] = static_cast<c_float>(u(i));
  }

  csc* P = csc_matrix(nU, nU, static_cast<c_int>(P_x.size()), P_x.data(), P_i.data(),
                      P_p.data());
  csc* A = csc_matrix(static_cast<c_int>(Acon.rows()), nU, static_cast<c_int>(A_x.size()),
                      A_x.data(), A_i.data(), A_p.data());

  OSQPData* data = static_cast<OSQPData*>(c_malloc(sizeof(OSQPData)));
  data->n = nU;
  data->m = static_cast<c_int>(Acon.rows());
  data->P = P;
  data->q = q_c.data();
  data->A = A;
  data->l = l_c.data();
  data->u = u_c.data();

  OSQPSettings* settings = static_cast<OSQPSettings*>(c_malloc(sizeof(OSQPSettings)));
  osqp_set_default_settings(settings);
  settings->verbose = 0;
  settings->warm_start = 1;
  settings->polish = 0;
  settings->eps_abs = 1e-3;
  settings->eps_rel = 1e-3;
  settings->max_iter = 4000;

  OSQPWorkspace* work = nullptr;
  const c_int setup_exit = osqp_setup(&work, data, settings);
  Input u0{0.0, 0.0};
  if (setup_exit != 0 || !work) {
    if (P) c_free(P);
    if (A) c_free(A);
    if (data) c_free(data);
    if (settings) c_free(settings);
    return u0;
  }

  std::vector<c_float> warm(nU);
  for (int i = 0; i < nU; ++i) warm[i] = static_cast<c_float>(u_warm_[i]);
  osqp_warm_start_x(work, warm.data());
  osqp_solve(work);

  out.iter = work->info->iter;
#ifdef PROFILING
  out.solve_time_s = work->info->solve_time;
#else
  out.solve_time_s = 0.0;
#endif
  if (work->info->status_val == OSQP_SOLVED ||
      work->info->status_val == OSQP_SOLVED_INACCURATE) {
    out.ok = true;
    for (int i = 0; i < nU; ++i) {
      out.u_seq[i] = work->solution->x[i];
      u_warm_[i] = out.u_seq[i];
    }
    u0[0] = out.u_seq[0];
    u0[1] = out.u_seq[1];
  }

  osqp_cleanup(work);
  if (P) c_free(P);
  if (A) c_free(A);
  if (data) c_free(data);
  if (settings) c_free(settings);
  return u0;
}

}  // namespace safe_corridor
