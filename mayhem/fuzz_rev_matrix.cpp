// Fuzz Stan Math's reverse-mode autodiff over matrix functions (Eigen<var> and var_value<Matrix>),
// then run the reverse sweep so both the value and the adjoint code paths execute.
#include <stan/math/rev.hpp>
#include <exception>
#include "fuzz_common.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
namespace sm = stan::math;

template <bool kVarMat, typename Mat, typename Vec>
static sm::var body(int op, const Mat& a, const Mat& b, const Vec& v, const Mat& s, int k, double x) {
  switch (op) {
    case 0: return sm::determinant(a);
    case 1: return sm::log_determinant(a);
    case 2: return sm::sum(sm::inverse(a));
    case 3: return sm::sum(sm::cholesky_decompose(s));
    case 4: return sm::sum(sm::mdivide_left(a, b));
    case 5: return sm::sum(sm::mdivide_left_spd(s, b));
    case 6:
      if constexpr (kVarMat) return sm::sum(a);
      else return sm::sum(sm::matrix_exp(a));
    case 7: return sm::sum(sm::eigenvalues_sym(s));
    case 8: return sm::sum(sm::singular_values(a));
    case 9: return sm::trace_quad_form(a, b);
    case 10: return sm::log_sum_exp(v) + sm::sum(sm::softmax(v)) + sm::sum(sm::log_softmax(v));
    case 11: return sm::sum(sm::multiply(a, b)) + sm::sum(sm::tcrossprod(a));
    case 12: return sm::sum(sm::generalized_inverse(a));
    case 13:
      if constexpr (kVarMat) return sm::sum(a);
      else return sm::sum(sm::mdivide_left_tri_low(a, b));
    case 14: return sm::sum(sm::matrix_power(a, k - 3));
    case 15: return sm::sum(sm::eigenvectors_sym(s));
    case 16: return sm::sum(sm::columns_dot_product(a, b)) + sm::sum(sm::rows_dot_self(a));
    case 17: return sm::sum(sm::cumulative_sum(v)) + sm::dot_self(v);
    case 18:
      if constexpr (kVarMat) return sm::sum(a);
      else return sm::sum(sm::quad_form_sym(s, b));
    case 19: return sm::log_determinant_spd(s);
    case 20: return sm::sum(sm::diag_pre_multiply(v, a));
    case 21:
      if constexpr (kVarMat) return sm::sum(a);
      else return sm::sum(sm::matrix_exp_multiply(a, b));
    case 22: return sm::sum(sm::svd_U(a)) + sm::sum(sm::svd_V(a));
    case 23: return sm::sum(sm::exp(a)) + sm::sum(sm::log1p_exp(a)) + sm::sum(sm::inv_logit(a));
    case 24: return sm::sum(sm::lgamma(s)) + sm::sum(sm::digamma(s)) + sm::sum(sm::inv_Phi(sm::inv_logit(s)));
    case 25:
      if constexpr (kVarMat) return sm::sum(a);
      else return sm::sum(sm::add_diag(a, x)) + sm::trace(a) + sm::prod(v);
    case 26:
      if constexpr (kVarMat) return sm::sum(a);
      else return sm::variance(v) + sm::sd(v) + sm::mean(v);
    case 27:
      if constexpr (kVarMat) return sm::sum(a);
      else return sm::sum(sm::scale_matrix_exp_multiply(x, a, b));
    default:
      if constexpr (kVarMat) return sm::sum(a);
      else return sm::sum(sm::mdivide_right(sm::transpose(b), a));
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size < 3) return -1;
  mfuzz::Reader rd(data, size);
  sm::nested_rev_autodiff nested;
  try {
    int op = rd.u8() % 29;
    bool vm = rd.flag();
    int k = rd.range(1, 5);
    MatrixXd a = rd.matrix(k, k), b = rd.matrix(k, k);
    MatrixXd s = mfuzz::spd(rd, k);
    VectorXd v = rd.vector(k);
    double x = rd.fin(4.0);
    if (!vm) {
      Eigen::Matrix<sm::var, -1, -1> A = sm::to_var(a), B = sm::to_var(b), S = sm::to_var(s);
      Eigen::Matrix<sm::var, -1, 1> V = sm::to_var(v);
      sm::var f = body<false>(op, A, B, V, S, k, x);
      f.grad();
    } else {
      sm::var_value<MatrixXd> A(a), B(b), S(s);
      sm::var_value<VectorXd> V(v);
      sm::var f = body<true>(op, A, B, V, S, k, x);
      f.grad();
    }
  } catch (const std::exception&) {
  }
  return 0;
}
