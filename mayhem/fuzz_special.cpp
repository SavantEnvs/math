// Fuzz Stan Math's scalar special functions at every autodiff level: double, var, fvar<double>
// and fvar<var> (forward-over-reverse), then run the sweeps that produce first/second derivatives.
#include <stan/math/mix.hpp>
#include <exception>
#include <vector>
#include "fuzz_common.hpp"

namespace sm = stan::math;
using sm::var;
using sm::fvar;

namespace {

double val_of(double x) { return x; }
double val_of(const var& x) { return x.val(); }
template <typename T>
double val_of(const fvar<T>& x) { return val_of(x.val_); }

template <typename T>
T lift(double x, double tangent) {
  if constexpr (std::is_same_v<T, double>) return x;
  else if constexpr (std::is_same_v<T, var>) return var(x);
  else if constexpr (std::is_same_v<T, fvar<double>>) return fvar<double>(x, tangent);
  else return fvar<var>(var(x), var(tangent));
}

template <typename T>
T special(int op, mfuzz::Reader& rd) {
  auto R = [&]() { return lift<T>((rd.u8() & 7) ? rd.fin(30.0) : rd.dbl(), (rd.u8() & 3) ? 1.0 : rd.fin(3.0)); };
  auto P = [&]() { return lift<T>((rd.u8() & 7) ? rd.pos(30.0) + 1e-4 : rd.dbl(), 1.0); };
  auto U = [&]() { return lift<T>((rd.u8() & 7) ? (rd.u16() + 0.5) / 65536.0 : rd.dbl(), 1.0); };
  auto N = [&](int hi = 12) { return rd.range(0, hi); };
  switch (op) {
    case 0: return sm::lgamma(R()) + sm::digamma(R()) + sm::trigamma(R());
    case 1: return sm::log1p_exp(R()) + sm::log1m_exp(R()) + sm::log1m(R()) + sm::log1p(R());
    case 2: return sm::erf(R()) + sm::erfc(R()) + sm::Phi(R()) + sm::Phi_approx(R());
    case 3: return sm::inv_Phi(U()) + sm::std_normal_log_qf(R());
    case 4: return sm::inv_cloglog(R()) + sm::log_inv_logit(R()) + sm::log1m_inv_logit(R()) + sm::logit(U());
    case 5: return sm::lambert_w0(R()) + sm::lambert_wm1(R());
    case 6: return sm::gamma_p(P(), P()) + sm::gamma_q(P(), P());
    case 7: return sm::lbeta(P(), P()) + sm::beta(P(), P()) + sm::binomial_coefficient_log(P(), P());
    case 8: return sm::log_diff_exp(R(), R()) + sm::log_sum_exp(R(), R()) + sm::log_inv_logit_diff(R(), R());
    case 9: return sm::lmultiply(R(), R()) + sm::multiply_log(R(), R());
    case 10: return sm::falling_factorial(P(), N()) + sm::rising_factorial(P(), N()) + sm::log_falling_factorial(P(), P()) + sm::log_rising_factorial(P(), P());
    case 11: return sm::owens_t(R(), R());
    case 12: return sm::bessel_first_kind(N(), R()) + sm::bessel_second_kind(N(), P());
    case 13: return sm::modified_bessel_first_kind(N(), R()) + sm::modified_bessel_second_kind(N(), P());
    case 14: return sm::log_modified_bessel_first_kind(P(), P());
    case 15: return sm::inc_beta(P(), P(), U());
    case 16: return sm::inv_inc_beta(P(), P(), U());
    case 17: return sm::fma(R(), R(), R()) + sm::hypot(R(), R()) + sm::fdim(R(), R()) + sm::atan2(R(), R());
    case 18: return sm::pow(R(), R()) + sm::fmod(R(), R()) + sm::cbrt(R()) + sm::expm1(R());
    case 19: return sm::hypergeometric_2F1(R(), R(), R(), U());
    case 20: return sm::hypergeometric_3F2(std::vector<T>{R(), R(), R()}, std::vector<T>{R(), R()}, U());
    case 21: return sm::hypergeometric_pFq(std::vector<T>{R(), R()}, std::vector<T>{R(), R()}, U());
    case 22: return sm::hypergeometric_1F0(R(), U());
    case 23: return sm::log_mix(U(), R(), R());
    case 24: return sm::binary_log_loss(N(1), U());
    case 25: return sm::lmgamma(N(5), P()) + sm::choose(N(30), N(30)) * T(1.0);
    case 26: return sm::log_inv_logit(R()) + sm::inv_square(R()) + sm::inv_sqrt(R()) + sm::inv(R()) + sm::inv_erfc(U());
    case 27: return sm::tgamma(R()) + sm::log_softmax(Eigen::Matrix<T, -1, 1>::Constant(3, R()))(0);
    case 28: return sm::hypergeometric_2F1(R(), R(), R(), R()) + sm::hypergeometric_3F2(std::vector<T>{R(), R(), R()}, std::vector<T>{R(), R()}, R());
    case 29: return sm::inc_beta(R(), R(), R()) + sm::log_modified_bessel_first_kind(R(), R());
    case 30: return sm::inv_Phi(U()) + sm::inv_cloglog(R()) + sm::log_diff_exp(R(), R());
    case 31: return sm::loglogistic_cdf(P(), P(), P()) * T(1.0) + sm::hypergeometric_1F0(R(), R());
    default: return sm::exp(R()) + sm::log(R()) + sm::sqrt(R()) + sm::tanh(R()) + sm::asinh(R()) + sm::acosh(R()) + sm::atanh(R());
  }
}

template <typename T>
void run(int op, mfuzz::Reader& rd) {
  sm::nested_rev_autodiff nested;
  T f = special<T>(op, rd);
  if constexpr (std::is_same_v<T, var>) {
    f.grad();
  } else if constexpr (std::is_same_v<T, fvar<var>>) {
    f.val_.grad();
    sm::set_zero_all_adjoints_nested();
    f.d_.grad();
  } else {
    (void)val_of(f);
  }
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size < 4) return -1;
  mfuzz::Reader rd(data, size);
  try {
    int op = rd.u8() % 33;
    int level = rd.u8() % 4;
    switch (level) {
      case 0: run<double>(op, rd); break;
      case 1: run<var>(op, rd); break;
      case 2: run<fvar<double>>(op, rd); break;
      default: run<fvar<var>>(op, rd); break;
    }
  } catch (const std::exception&) {
  }
  return 0;
}
