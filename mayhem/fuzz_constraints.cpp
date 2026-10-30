// Fuzz Stan Math's constraining/unconstraining transforms (with and without the log-Jacobian) in
// reverse mode, including the constrain -> free round trip.
#include <stan/math/rev.hpp>
#include <exception>
#include <vector>
#include "fuzz_common.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
namespace sm = stan::math;
using sm::var;

template <typename T>
static var accumulate(const T& x, var lp) {
  return sm::sum(x) + lp;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size < 4) return -1;
  mfuzz::Reader rd(data, size);
  sm::nested_rev_autodiff nested;
  try {
    int op = rd.u8() % 24;
    bool jac = rd.flag();
    int k = rd.range(1, 6);
    int m = rd.range(1, 5);
    VectorXd raw = rd.vector(k);
    VectorXd rawf = rd.fvector(k, 10.0);
    Eigen::Matrix<var, -1, 1> X = sm::to_var(raw);
    Eigen::Matrix<var, -1, 1> Xf = sm::to_var(rawf);
    var lp = 0;
    var lb = rd.dbl(), ub = rd.dbl();
    var f = 0;
    switch (op) {
      case 0: { var y = jac ? sm::lb_constrain(X(0), lb, lp) : sm::lb_constrain(X(0), lb); f = y + lp; sm::lb_free(y.val(), lb.val()); break; }
      case 1: { var y = jac ? sm::ub_constrain(X(0), ub, lp) : sm::ub_constrain(X(0), ub); f = y + lp; sm::ub_free(y.val(), ub.val()); break; }
      case 2: { var y = jac ? sm::lub_constrain(X(0), lb, ub, lp) : sm::lub_constrain(X(0), lb, ub); f = y + lp; sm::lub_free(y.val(), lb.val(), ub.val()); break; }
      case 3: { var y = jac ? sm::offset_multiplier_constrain(X(0), lb, ub, lp) : sm::offset_multiplier_constrain(X(0), lb, ub); f = y + lp; sm::offset_multiplier_free(y.val(), lb.val(), ub.val()); break; }
      case 4: { auto y = jac ? sm::ordered_constrain(X, lp) : sm::ordered_constrain(X); f = accumulate(y, lp); sm::ordered_free(sm::value_of(y)); break; }
      case 5: { auto y = jac ? sm::positive_ordered_constrain(X, lp) : sm::positive_ordered_constrain(X); f = accumulate(y, lp); sm::positive_ordered_free(sm::value_of(y)); break; }
      case 6: { auto y = jac ? sm::simplex_constrain(Xf, lp) : sm::simplex_constrain(Xf); f = accumulate(y, lp); sm::simplex_free(sm::value_of(y)); break; }
      case 7: { auto y = jac ? sm::unit_vector_constrain(X, lp) : sm::unit_vector_constrain(X); f = accumulate(y, lp); sm::unit_vector_free(sm::value_of(y)); break; }
      case 8: { auto y = sm::sum_to_zero_constrain(X); f = sm::sum(y); sm::sum_to_zero_free(sm::value_of(y)); break; }
      case 9: { var y = jac ? sm::prob_constrain(X(0), lp) : sm::prob_constrain(X(0)); f = y + lp; sm::prob_free(y.val()); break; }
      case 10: { var y = jac ? sm::corr_constrain(X(0), lp) : sm::corr_constrain(X(0)); f = y + lp; sm::corr_free(y.val()); break; }
      case 11: { var y = jac ? sm::positive_constrain(X(0), lp) : sm::positive_constrain(X(0)); f = y + lp; sm::positive_free(y.val()); break; }
      case 12: { int n = m * (m + 1) / 2; VectorXd v = rd.fvector(n, 10.0); Eigen::Matrix<var, -1, 1> V = sm::to_var(v); auto L = jac ? sm::cholesky_factor_constrain(V, m, m, lp) : sm::cholesky_factor_constrain(V, m, m); f = accumulate(L, lp); sm::cholesky_factor_free(sm::value_of(L)); break; }
      case 13: { int n = m * (m - 1) / 2; if (n == 0) break; VectorXd v = rd.fvector(n, 10.0); Eigen::Matrix<var, -1, 1> V = sm::to_var(v); auto L = jac ? sm::cholesky_corr_constrain(V, m, lp) : sm::cholesky_corr_constrain(V, m); f = accumulate(L, lp); sm::cholesky_corr_free(sm::value_of(L)); break; }
      case 14: { int n = m * (m - 1) / 2; if (n == 0) break; VectorXd v = rd.fvector(n, 10.0); Eigen::Matrix<var, -1, 1> V = sm::to_var(v); auto C = jac ? sm::corr_matrix_constrain(V, m, lp) : sm::corr_matrix_constrain(V, m); f = accumulate(C, lp); sm::corr_matrix_free(sm::value_of(C)); break; }
      case 15: { int n = m + m * (m - 1) / 2; VectorXd v = rd.fvector(n, 10.0); Eigen::Matrix<var, -1, 1> V = sm::to_var(v); auto C = jac ? sm::cov_matrix_constrain(V, m, lp) : sm::cov_matrix_constrain(V, m); f = accumulate(C, lp); sm::cov_matrix_free(sm::value_of(C)); break; }
      case 16: { int n = m * (m - 1) / 2 + m; VectorXd v = rd.fvector(n, 10.0); Eigen::Matrix<var, -1, 1> V = sm::to_var(v); auto C = jac ? sm::cov_matrix_constrain_lkj(V, m, lp) : sm::cov_matrix_constrain_lkj(V, m); f = accumulate(C, lp); break; }
      case 17: { MatrixXd a = rd.fmatrix(m, k, 10.0); Eigen::Matrix<var, -1, -1> A = sm::to_var(a); Eigen::Matrix<var, -1, -1> y; if (jac) y = sm::stochastic_column_constrain(A, lp); else y = sm::stochastic_column_constrain(A); f = accumulate(y, lp); sm::stochastic_column_free(sm::value_of(y)); break; }
      case 18: { MatrixXd a = rd.fmatrix(k, m, 10.0); Eigen::Matrix<var, -1, -1> A = sm::to_var(a); Eigen::Matrix<var, -1, -1> y; if (jac) y = sm::stochastic_row_constrain(A, lp); else y = sm::stochastic_row_constrain(A); f = accumulate(y, lp); sm::stochastic_row_free(sm::value_of(y)); break; }
      case 19: { std::vector<var> xs(X.data(), X.data() + k); auto y = jac ? sm::lub_constrain(xs, lb, ub, lp) : sm::lub_constrain(xs, lb, ub); f = sm::sum(y) + lp; break; }
      case 20: { MatrixXd a = rd.fmatrix(m, k, 10.0); Eigen::Matrix<var, -1, -1> A = sm::to_var(a); auto y = sm::sum_to_zero_constrain(A); f = sm::sum(y); sm::sum_to_zero_free(sm::value_of(y)); break; }
      case 21: { var y = jac ? sm::lb_constrain(X(0), -std::numeric_limits<double>::infinity(), lp) : sm::ub_constrain(X(0), std::numeric_limits<double>::infinity()); f = y + lp; break; }
      case 22: { double v0 = rd.dbl(); double y = sm::lub_constrain(v0, lb.val(), ub.val()); double z = sm::lub_free(y, lb.val(), ub.val()); f = y + z; break; }
      default: { auto y = sm::inv_logit(X); f = sm::sum(y) + sm::sum(sm::logit(y)) + sm::sum(sm::log1m_inv_logit(X)); break; }
    }
    f.grad();
  } catch (const std::exception&) {
  }
  return 0;
}
