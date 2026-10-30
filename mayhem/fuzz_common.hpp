// Shared helpers for the Stan Math fuzz harnesses: a bounded byte reader that turns fuzzer
// bytes into dimensions, selectors and doubles (including NaN/inf/denormal edge values).
#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <limits>
#include <Eigen/Dense>

namespace mfuzz {

struct Reader {
  const uint8_t* p;
  size_t n;
  Reader(const uint8_t* d, size_t s) : p(d), n(s) {}
  bool empty() const { return n == 0; }
  uint8_t u8() {
    if (n == 0) return 0;
    --n;
    return *p++;
  }
  uint16_t u16() {
    uint16_t a = u8();
    return static_cast<uint16_t>(a | (static_cast<uint16_t>(u8()) << 8));
  }
  int range(int lo, int hi) { return lo + static_cast<int>(u8() % (hi - lo + 1)); }
  bool flag() { return (u8() & 1) != 0; }
  double dbl() {
    uint8_t s = u8();
    switch (s & 7) {
      case 5:
        return static_cast<double>(u16()) / 1024.0;
      case 6: {
        static const double special[] = {0.0, -0.0, 1.0, -1.0, 0.5, 2.0,
          std::numeric_limits<double>::infinity(),
          -std::numeric_limits<double>::infinity(),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::max(),
          std::numeric_limits<double>::min(),
          std::numeric_limits<double>::denorm_min(), 1e-300, 1e300, 1e-8, 1e8};
        return special[u8() % (sizeof(special) / sizeof(special[0]))];
      }
      case 7: {
        uint64_t bits = 0;
        for (int i = 0; i < 8; ++i) bits = (bits << 8) | u8();
        double d;
        std::memcpy(&d, &bits, sizeof d);
        return d;
      }
      default:
        return static_cast<double>(static_cast<int16_t>(u16())) / 256.0;
    }
  }
  // A finite, moderately sized double (for arguments where only domain errors, not
  // overflow in the harness itself, are interesting).
  double fin(double lim = 1e3) {
    double d = dbl();
    if (!std::isfinite(d)) return 0.0;
    if (d > lim) return lim;
    if (d < -lim) return -lim;
    return d;
  }
  double pos(double lim = 1e3) {
    double d = std::fabs(fin(lim));
    return d;
  }
  Eigen::MatrixXd matrix(int r, int c) {
    Eigen::MatrixXd m(r, c);
    for (int j = 0; j < c; ++j)
      for (int i = 0; i < r; ++i) m(i, j) = dbl();
    return m;
  }
  Eigen::MatrixXd fmatrix(int r, int c, double lim = 1e3) {
    Eigen::MatrixXd m(r, c);
    for (int j = 0; j < c; ++j)
      for (int i = 0; i < r; ++i) m(i, j) = fin(lim);
    return m;
  }
  Eigen::VectorXd vector(int n_) {
    Eigen::VectorXd v(n_);
    for (int i = 0; i < n_; ++i) v(i) = dbl();
    return v;
  }
  Eigen::VectorXd fvector(int n_, double lim = 1e3) {
    Eigen::VectorXd v(n_);
    for (int i = 0; i < n_; ++i) v(i) = fin(lim);
    return v;
  }
};

// Symmetric positive-definite matrix built from finite entries: A*A' + eps*I.
inline Eigen::MatrixXd spd(Reader& rd, int k) {
  Eigen::MatrixXd a = rd.fmatrix(k, k, 10.0);
  Eigen::MatrixXd s = a * a.transpose();
  s.diagonal().array() += 1e-3;
  return s;
}

}  // namespace mfuzz
