// Fuzz Stan Math's probability functions (lpdf/lpmf/lcdf/rng) with reverse-mode variables, then
// run the gradient sweep.
#include <stan/math/rev.hpp>
#include <boost/random/additive_combine.hpp>
#include <exception>
#include <vector>
#include "fuzz_common.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
namespace sm = stan::math;
using sm::var;

namespace {

struct Gen {
  mfuzz::Reader& rd;
  double real() { return (rd.u8() & 7) ? rd.fin(50.0) : rd.dbl(); }
  double posd() { return (rd.u8() & 7) ? rd.pos(50.0) + 1e-3 : rd.dbl(); }
  double unit() { return (rd.u8() & 7) ? (rd.u16() + 0.5) / 65536.0 : rd.dbl(); }
  var r() { return var(real()); }
  var p() { return var(posd()); }
  var u() { return var(unit()); }
  int nn(int hi = 40) { return rd.range(0, hi); }
  VectorXd rv(int k) { VectorXd v(k); for (int i = 0; i < k; ++i) v(i) = real(); return v; }
  VectorXd simplex(int k) {
    VectorXd v(k);
    double t = 0;
    for (int i = 0; i < k; ++i) { v(i) = posd() + 1e-3; t += v(i); }
    v /= t;
    return v;
  }
  MatrixXd spd(int k) { return mfuzz::spd(rd, k); }
};

var dist(int op, Gen& g, boost::ecuyer1988& rng) {
  mfuzz::Reader& rd = g.rd;
  int k = rd.range(1, 4);
  switch (op) {
    case 0: return sm::normal_lpdf(g.r(), g.r(), g.p());
    case 1: return sm::student_t_lpdf(g.r(), g.p(), g.r(), g.p());
    case 2: return sm::gamma_lpdf(g.p(), g.p(), g.p());
    case 3: return sm::beta_lpdf(g.u(), g.p(), g.p());
    case 4: return sm::cauchy_lpdf(g.r(), g.r(), g.p());
    case 5: return sm::lognormal_lpdf(g.p(), g.r(), g.p());
    case 6: return sm::exp_mod_normal_lpdf(g.r(), g.r(), g.p(), g.p());
    case 7: return sm::skew_normal_lpdf(g.r(), g.r(), g.p(), g.r());
    case 8: return sm::skew_double_exponential_lpdf(g.r(), g.r(), g.p(), g.u());
    case 9: return sm::von_mises_lpdf(g.r(), g.r(), g.p());
    case 10: return sm::gumbel_lpdf(g.r(), g.r(), g.p());
    case 11: return sm::logistic_lpdf(g.r(), g.r(), g.p());
    case 12: return sm::weibull_lpdf(g.p(), g.p(), g.p());
    case 13: return sm::frechet_lpdf(g.p(), g.p(), g.p());
    case 14: return sm::pareto_type_2_lpdf(g.p(), g.r(), g.p(), g.p());
    case 15: return sm::inv_gamma_lpdf(g.p(), g.p(), g.p());
    case 16: return sm::chi_square_lpdf(g.p(), g.p()) + sm::inv_chi_square_lpdf(g.p(), g.p());
    case 17: return sm::scaled_inv_chi_square_lpdf(g.p(), g.p(), g.p());
    case 18: return sm::rayleigh_lpdf(g.p(), g.p()) + sm::double_exponential_lpdf(g.r(), g.r(), g.p());
    case 19: return sm::wiener_lpdf(g.p(), g.p(), g.p(), g.u(), g.r());
    case 20: return sm::loglogistic_lpdf(g.p(), g.p(), g.p());
    case 21: return sm::beta_proportion_lpdf(g.u(), g.u(), g.p());
    case 22: return sm::normal_lcdf(g.r(), g.r(), g.p()) + sm::std_normal_lcdf(g.r());
    case 23: return sm::gamma_lcdf(g.p(), g.p(), g.p()) + sm::gamma_lccdf(g.p(), g.p(), g.p());
    case 24: return sm::beta_lcdf(g.u(), g.p(), g.p()) + sm::beta_lccdf(g.u(), g.p(), g.p());
    case 25: return sm::student_t_lcdf(g.r(), g.p(), g.r(), g.p()) + sm::student_t_lccdf(g.r(), g.p(), g.r(), g.p());
    case 26: return sm::wiener_lpdf(g.p(), g.p(), g.p(), g.u(), g.r(), g.p(), g.u(), g.p());
    case 27: return sm::poisson_lpmf(g.nn(), g.p()) + sm::poisson_log_lpmf(g.nn(), g.r());
    case 28: { int n = g.nn(); return sm::binomial_lpmf(n, n + g.nn(), g.u()) + sm::binomial_logit_lpmf(n, n + g.nn(), g.r()); }
    case 29: { int n = g.nn(); return sm::beta_binomial_lpmf(n, n + g.nn(), g.p(), g.p()); }
    case 30: return sm::neg_binomial_lpmf(g.nn(), g.p(), g.p()) + sm::neg_binomial_2_lpmf(g.nn(), g.p(), g.p());
    case 31: return sm::neg_binomial_2_log_lpmf(g.nn(), g.r(), g.p());
    case 32: return sm::bernoulli_lpmf(g.nn(1), g.u()) + sm::bernoulli_logit_lpmf(g.nn(1), g.r());
    case 33: { VectorXd b = g.rv(k); return sm::categorical_logit_lpmf(rd.range(1, k), b) + sm::categorical_lpmf(rd.range(1, k), g.simplex(k)); }
    case 34: { VectorXd c = g.rv(k); std::sort(c.data(), c.data() + k); return sm::ordered_logistic_lpmf(rd.range(1, k + 1), g.r(), c) + sm::ordered_probit_lpmf(rd.range(1, k + 1), g.r(), c); }
    case 35: { int N = g.nn(30), a = g.nn(30), b = g.nn(30); return sm::hypergeometric_lpmf(g.nn(10), N, a, b) + var(0); }
    case 36: return sm::beta_neg_binomial_lpmf(g.nn(), g.p(), g.p(), g.p());
    case 37: { MatrixXd S = g.spd(k); VectorXd y = g.rv(k), mu = g.rv(k); return sm::multi_normal_lpdf(y, mu, S) + sm::multi_normal_prec_lpdf(y, mu, S); }
    case 38: { MatrixXd L = Eigen::LLT<MatrixXd>(g.spd(k)).matrixL(); VectorXd y = g.rv(k), mu = g.rv(k); return sm::multi_normal_cholesky_lpdf(y, mu, L); }
    case 39: { MatrixXd S = g.spd(k); VectorXd y = g.rv(k), mu = g.rv(k); return sm::multi_student_t_lpdf(y, g.p(), mu, S); }
    case 40: { MatrixXd S = g.spd(k); VectorXd y = g.rv(k), mu = g.rv(k); MatrixXd L = Eigen::LLT<MatrixXd>(S).matrixL(); return sm::multi_student_t_cholesky_lpdf(y, g.p(), mu, L); }
    case 41: { MatrixXd S = g.spd(k); MatrixXd y = rd.fmatrix(2, k); VectorXd w(2); w << g.posd(), g.posd(); return sm::multi_gp_lpdf(y, S, w); }
    case 42: { VectorXd th = g.simplex(k); VectorXd al(k); for (int i = 0; i < k; ++i) al(i) = g.posd(); return sm::dirichlet_lpdf(th, al); }
    case 43: { MatrixXd S = g.spd(k); Eigen::VectorXd d = S.diagonal().array().sqrt(); MatrixXd C = S.array() / (d * d.transpose()).array(); return sm::lkj_corr_lpdf(C, g.p()); }
    case 44: { MatrixXd S = g.spd(k); MatrixXd L = Eigen::LLT<MatrixXd>(S).matrixL(); Eigen::VectorXd d = S.diagonal().array().sqrt(); MatrixXd C = S.array() / (d * d.transpose()).array(); MatrixXd Lc = Eigen::LLT<MatrixXd>(C).matrixL(); return sm::lkj_corr_cholesky_lpdf(Lc, g.p()); }
    case 45: { MatrixXd W = g.spd(k), S = g.spd(k); return sm::wishart_lpdf(W, g.p() + k, S) + sm::inv_wishart_lpdf(W, g.p() + k, S); }
    case 46: { MatrixXd W = Eigen::LLT<MatrixXd>(g.spd(k)).matrixL(), S = Eigen::LLT<MatrixXd>(g.spd(k)).matrixL(); return sm::wishart_cholesky_lpdf(W, g.p() + k, S) + sm::inv_wishart_cholesky_lpdf(W, g.p() + k, S); }
    case 47: { std::vector<int> ns(k); for (auto& n : ns) n = g.nn(20); return sm::multinomial_lpmf(ns, g.simplex(k)) + sm::multinomial_logit_lpmf(ns, g.rv(k)); }
    case 48: { std::vector<int> ns(k); for (auto& n : ns) n = g.nn(20); VectorXd al(k); for (int i = 0; i < k; ++i) al(i) = g.posd(); return sm::dirichlet_multinomial_lpmf(ns, al); }
    case 49: { MatrixXd y = rd.fmatrix(2, k); MatrixXd Sigma = g.spd(2), D = g.spd(k); MatrixXd mu = rd.fmatrix(2, k); return sm::matrix_normal_prec_lpdf(y, mu, Sigma, D); }
    case 50: { MatrixXd F = rd.fmatrix(k, 2), G = rd.fmatrix(k, k), V = g.spd(2), W = g.spd(k); MatrixXd y = rd.fmatrix(2, 3); VectorXd m0 = g.rv(k); MatrixXd C0 = g.spd(k); return sm::gaussian_dlm_obs_lpdf(y, F, G, V, W, m0, C0); }
    case 51: { VectorXd y = g.rv(k), mu = g.rv(k), s = VectorXd::Ones(k) * g.posd(); return sm::normal_lpdf(y, mu, s) + sm::normal_lpdf(g.r(), mu, s) + sm::cauchy_lpdf(y, mu, g.p()); }
    case 52: { VectorXd mu = g.rv(k); std::vector<double> y(k); for (auto& e : y) e = g.real(); return sm::normal_lpdf(y, mu, g.p()) + sm::student_t_lpdf(y, g.p(), mu, g.p()) + sm::gamma_lpdf(y, g.p(), g.p()); }
    // random number generators (value only, no gradient)
    case 53: { sm::normal_rng(g.real(), g.posd(), rng); sm::student_t_rng(g.posd(), g.real(), g.posd(), rng); sm::gamma_rng(g.posd(), g.posd(), rng); sm::beta_rng(g.posd(), g.posd(), rng); sm::von_mises_rng(g.real(), g.posd(), rng); return var(0); }
    case 54: { sm::poisson_rng(g.posd(), rng); sm::binomial_rng(g.nn(), g.unit(), rng); sm::neg_binomial_2_rng(g.posd(), g.posd(), rng); sm::beta_binomial_rng(g.nn(), g.posd(), g.posd(), rng); return var(0); }
    case 55: { sm::multi_normal_rng(g.rv(k), g.spd(k), rng); sm::wishart_rng(g.posd() + k, g.spd(k), rng); sm::inv_wishart_rng(g.posd() + k, g.spd(k), rng); sm::lkj_corr_rng(k, g.posd(), rng); sm::dirichlet_rng(g.simplex(k), rng); return var(0); }
    case 56: { sm::categorical_logit_rng(g.rv(k), rng); sm::multinomial_rng(g.simplex(k), g.nn(), rng); sm::hypergeometric_rng(g.nn(20), g.nn(20), g.nn(20), rng); return var(0); }
    default: return sm::exponential_lpdf(g.p(), g.p()) + sm::uniform_lpdf(g.r(), g.r(), g.r()) + sm::std_normal_lpdf(g.r());
  }
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size < 4) return -1;
  mfuzz::Reader rd(data, size);
  sm::nested_rev_autodiff nested;
  try {
    int op = rd.u8() % 58;
    boost::ecuyer1988 rng(rd.u16());
    Gen g{rd};
    var f = dist(op, g, rng);
    f.grad();
  } catch (const std::exception&) {
  }
  return 0;
}
