// Fuzz Stan Math's higher-order functors that run header-only: the Boost-odeint ODE solvers
// (rk45, ckrk), 1-D quadrature, the Powell algebraic solver and adaptive cubature, with
// reverse-mode parameters so the coupled sensitivity systems and implicit-function gradients run.
#include <stan/math/rev.hpp>
#include <exception>
#include <vector>
#include "fuzz_common.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
namespace sm = stan::math;
using sm::var;

namespace {

struct Oscillator {
  template <typename T_t, typename T_y, typename T_a, typename T_b>
  inline Eigen::Matrix<stan::return_type_t<T_t, T_y, T_a, T_b>, -1, 1>
  operator()(const T_t& t, const Eigen::Matrix<T_y, -1, 1>& y, std::ostream*, const T_a& damp, const T_b& force) const {
    Eigen::Matrix<stan::return_type_t<T_t, T_y, T_a, T_b>, -1, 1> dy(2);
    dy(0) = y(1);
    dy(1) = -y(0) - damp * y(1) + force * sm::sin(t);
    return dy;
  }
};

struct VanDerPol {
  template <typename T_t, typename T_y, typename T_m>
  inline Eigen::Matrix<stan::return_type_t<T_t, T_y, T_m>, -1, 1>
  operator()(const T_t&, const Eigen::Matrix<T_y, -1, 1>& y, std::ostream*, const T_m& mu) const {
    Eigen::Matrix<stan::return_type_t<T_t, T_y, T_m>, -1, 1> dy(2);
    dy(0) = y(1);
    dy(1) = mu * (1 - y(0) * y(0)) * y(1) - y(0);
    return dy;
  }
};

struct Integrand {
  template <typename T_x, typename T_xc, typename T_a, typename T_b>
  inline stan::return_type_t<T_x, T_xc, T_a, T_b>
  operator()(const T_x& x, const T_xc&, std::ostream*, const T_a& a, const T_b& b) const {
    return sm::exp(-a * x * x) * sm::cos(b * x);
  }
};

struct Cubic {
  template <typename T_x, typename T_a, typename T_b>
  inline Eigen::Matrix<stan::return_type_t<T_x, T_a, T_b>, -1, 1>
  operator()(const Eigen::Matrix<T_x, -1, 1>& x, std::ostream*, const T_a& a, const T_b& b) const {
    Eigen::Matrix<stan::return_type_t<T_x, T_a, T_b>, -1, 1> z(2);
    z(0) = x(0) * x(0) + x(1) - a;
    z(1) = x(1) * x(1) * x(1) - x(0) * b;
    return z;
  }
};

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size < 8) return -1;
  mfuzz::Reader rd(data, size);
  sm::nested_rev_autodiff nested;
  try {
    int op = rd.u8() % 6;
    var f = 0;
    switch (op) {
      case 0:
      case 1: {
        VectorXd y0 = rd.fvector(2, 5.0);
        var damp = rd.fin(3.0), force = rd.fin(3.0);
        Eigen::Matrix<var, -1, 1> Y0 = sm::to_var(y0);
        std::vector<double> ts;
        double t = rd.fin(2.0);
        int nts = rd.range(1, 4);
        double t0 = t;
        for (int i = 0; i < nts; ++i) { t += rd.pos(3.0); ts.push_back(t); }
        double rtol = std::pow(10.0, -rd.range(2, 8)), atol = std::pow(10.0, -rd.range(2, 8));
        long max_steps = rd.range(1, 2000);
        std::vector<Eigen::Matrix<var, -1, 1>> out;
        if (op == 0) out = sm::ode_rk45_tol(Oscillator(), Y0, t0, ts, rtol, atol, max_steps, nullptr, damp, force);
        else out = sm::ode_ckrk_tol(Oscillator(), Y0, t0, ts, rtol, atol, max_steps, nullptr, damp, force);
        for (auto& o : out) f += sm::sum(o);
        break;
      }
      case 2: {
        VectorXd y0 = rd.fvector(2, 2.0);
        var mu = rd.pos(3.0);
        std::vector<var> ts{rd.pos(5.0) + 0.1};
        std::vector<Eigen::Matrix<var, -1, 1>> out = sm::ode_rk45(VanDerPol(), y0, 0.0, ts, nullptr, mu);
        for (auto& o : out) f += sm::sum(o);
        break;
      }
      case 3: {
        double a = rd.fin(20.0), b = rd.fin(20.0);
        var p = rd.pos(10.0), q = rd.fin(10.0);
        if (rd.flag()) a = -std::numeric_limits<double>::infinity();
        if (rd.flag()) b = std::numeric_limits<double>::infinity();
        f = sm::integrate_1d_impl(Integrand(), a, b, std::pow(10.0, -rd.range(1, 8)), nullptr, p, q);
        break;
      }
      case 4: {
        VectorXd guess = rd.fvector(2, 5.0);
        var a = rd.fin(10.0), b = rd.fin(10.0);
        auto sol = sm::solve_powell_tol(Cubic(), guess, std::pow(10.0, -rd.range(2, 10)),
                                            std::pow(10.0, -rd.range(2, 10)), rd.range(1, 500), nullptr, a, b);
        f = sm::sum(sol);
        break;
      }
      default: {
        int dim = rd.range(1, 3);
        VectorXd lo(dim), hi(dim);
        for (int i = 0; i < dim; ++i) { lo(i) = rd.fin(3.0); hi(i) = lo(i) + rd.pos(3.0); }
        Eigen::VectorXd p(1);
        p(0) = rd.pos(5.0);
        std::tuple<Eigen::VectorXd> pars(p);
        auto integrand = [](const auto& x, const auto& pr) { return std::exp(-pr(0) * sm::sum(sm::square(sm::as_array_or_scalar(x)))); };
        f = sm::hcubature(integrand, pars, dim, lo, hi, rd.range(1, 2000), 1e-6, 1e-6);
        break;
      }
    }
    f.grad();
  } catch (const std::exception&) {
  }
  return 0;
}
