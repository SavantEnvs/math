// Fuzz Stan Math's double-valued (prim) linear algebra and array functions.
#include <stan/math/prim.hpp>
#include <exception>
#include "fuzz_common.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;
namespace sm = stan::math;

static void run_op(mfuzz::Reader& rd) {
  int op = rd.u8() % 40;
  int r = rd.range(1, 6), c = rd.range(1, 6), k = rd.range(1, 6);
  switch (op) {
    case 0: { MatrixXd a = rd.matrix(k, k); sm::determinant(a); break; }
    case 1: { MatrixXd a = rd.matrix(k, k); sm::log_determinant(a); break; }
    case 2: { MatrixXd a = rd.matrix(k, k); sm::inverse(a); break; }
    case 3: { MatrixXd a = rd.matrix(k, k); sm::cholesky_decompose(a); break; }
    case 4: { MatrixXd a = mfuzz::spd(rd, k); sm::cholesky_decompose(a); sm::log_determinant_spd(a); break; }
    case 5: { MatrixXd a = rd.fmatrix(k, k, 4.0); sm::matrix_exp(a); break; }
    case 6: { MatrixXd a = rd.matrix(k, k), b = rd.matrix(k, c); sm::mdivide_left_tri_low(a, b); break; }
    case 7: { MatrixXd a = mfuzz::spd(rd, k), b = rd.matrix(k, c); sm::mdivide_left_spd(a, b); break; }
    case 8: { MatrixXd a = rd.fmatrix(k, k); sm::eigenvalues_sym(a); sm::eigenvectors_sym(a); break; }
    case 9: { MatrixXd a = rd.fmatrix(r, c); sm::singular_values(a); sm::svd_U(a); sm::svd_V(a); break; }
    case 10: { MatrixXd a = rd.matrix(r, c); sm::tcrossprod(a); sm::crossprod(a); break; }
    case 11: { MatrixXd a = rd.fmatrix(k, k), b = rd.fmatrix(k, c); sm::quad_form(a, b); break; }
    case 12: { MatrixXd a = rd.fmatrix(k, k); VectorXd v = rd.fvector(k); sm::quad_form(a, v); sm::trace_quad_form(a, a); break; }
    case 13: { VectorXd v = rd.vector(k); sm::softmax(v); sm::log_softmax(v); sm::log_sum_exp(v); break; }
    case 14: { VectorXd v = rd.vector(k); sm::cumulative_sum(v); sm::sort_asc(v); sm::sort_desc(v); sm::sort_indices_asc(v); break; }
    case 15: { VectorXd v = rd.vector(k); sm::rank(v, rd.range(0, k + 1)); break; }
    case 16: { MatrixXd a = rd.fmatrix(k, k, 3.0); sm::matrix_power(a, rd.range(-2, 6)); break; }
    case 17: { MatrixXd a = rd.fmatrix(r, c); sm::generalized_inverse(a); break; }
    case 18: { MatrixXd a = rd.fmatrix(r, c); sm::qr_thin_Q(a); sm::qr_thin_R(a); sm::qr_Q(a); sm::qr_R(a); break; }
    case 19: { MatrixXd a = rd.fmatrix(k, k); sm::eigenvalues(a); sm::eigenvectors(a); break; }
    case 20: { MatrixXd a = rd.fmatrix(k, k); sm::complex_schur_decompose_t(a); sm::complex_schur_decompose_u(a); break; }
    case 21: { VectorXd v = rd.fvector(k); sm::fft(sm::to_complex(v, VectorXd::Zero(k))); break; }
    case 22: { MatrixXd a = rd.fmatrix(r, c); sm::fft2(sm::to_complex(a, MatrixXd::Zero(r, c))); break; }
    case 23: { MatrixXd a = rd.fmatrix(k, k, 3.0); MatrixXd b = rd.fmatrix(k, c, 3.0); sm::matrix_exp_multiply(a, b); sm::scale_matrix_exp_multiply(rd.fin(2.0), a, b); break; }
    case 24: { MatrixXd a = rd.matrix(k, k); sm::trace(a); sm::sum(a); sm::prod(a); sm::mean(a); sm::variance(a); sm::sd(a); break; }
    case 25: { MatrixXd a = rd.matrix(r, c); sm::columns_dot_self(a); sm::rows_dot_self(a); sm::columns_dot_product(a, a); sm::rows_dot_product(a, a); break; }
    case 26: { MatrixXd a = rd.matrix(r, c); sm::rep_matrix(a.col(0).eval(), rd.range(0, 5)); sm::append_row(a, a); sm::append_col(a, a); break; }
    case 27: { VectorXd v = rd.vector(k); sm::head(v, rd.range(0, k + 1)); sm::tail(v, rd.range(0, k + 1)); sm::segment(v, rd.range(0, k + 1), rd.range(0, k + 1)); break; }
    case 28: { VectorXd v = rd.vector(k); sm::reverse(v); sm::log_softmax(v); sm::cumulative_sum(v); sm::norm1(v); sm::norm2(v); sm::dot_self(v); break; }
    case 29: { MatrixXd a = mfuzz::spd(rd, k); VectorXd v = rd.fvector(k); sm::trace_gen_quad_form(a, a, a); sm::quad_form_sym(a, v); sm::multiply_lower_tri_self_transpose(a); break; }
    case 30: { MatrixXd a = rd.fmatrix(k, k); sm::singular_values(a); { MatrixXd s = mfuzz::spd(rd, k); auto ldlt = sm::make_ldlt_factor(s); sm::log_determinant_ldlt(ldlt); sm::trace_inv_quad_form_ldlt(ldlt, s); } break; }
    case 31: { VectorXd v = rd.vector(k); sm::log_mix(v, v); sm::variance(v); sm::log_sum_exp(v, rd.dbl()); break; }
    case 32: { MatrixXd a = rd.fmatrix(k, k); sm::add_diag(a, rd.fin()); sm::diag_pre_multiply(a.col(0).eval(), a); sm::diag_post_multiply(a, a.col(0).eval()); break; }
    case 33: { std::vector<double> x(rd.range(0, 8)); for (auto& e : x) e = rd.dbl(); sm::sort_asc(x); sm::log_sum_exp(x); sm::cumulative_sum(x); break; }
    case 34: { std::vector<double> x(rd.range(0, 8)); for (auto& e : x) e = rd.dbl(); sm::to_vector(x); sm::to_array_1d(sm::to_vector(x)); sm::to_matrix(x, rd.range(0, 4), rd.range(0, 4)); break; }
    case 35: { MatrixXd a = rd.matrix(r, c); sm::cov_exp_quad(std::vector<double>{rd.fin(), rd.fin()}, rd.pos(), rd.pos()); sm::to_vector(a); break; }
    case 36: { VectorXd v = rd.fvector(k); std::vector<VectorXd> xs{v, v + v}; sm::gp_exp_quad_cov(xs, rd.pos(), rd.pos()); sm::gp_matern32_cov(xs, rd.pos(), rd.pos()); sm::gp_periodic_cov(xs, rd.pos(), rd.pos(), rd.pos()); break; }
    case 37: { MatrixXd a = rd.fmatrix(k, k); sm::inv_square(a); sm::inv_logit(a); sm::log1p_exp(a); sm::lgamma(a); sm::digamma(a); break; }
    case 38: { MatrixXd a = rd.fmatrix(k, k); sm::cholesky_decompose(sm::symmetrize_from_lower_tri(a)); break; }
    case 39: { MatrixXd a = rd.fmatrix(k, k, 5.0); VectorXd b = rd.fvector(k, 5.0); sm::mdivide_right_tri_low(b.transpose().eval(), a); sm::mdivide_left(a, b); sm::mdivide_right(b.transpose().eval(), a); break; }
  }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size < 2) return -1;
  mfuzz::Reader rd(data, size);
  try {
    int nops = 1 + rd.u8() % 3;
    for (int i = 0; i < nops && !rd.empty(); ++i) run_op(rd);
  } catch (const std::exception&) {
  }
  return 0;
}
