# check_2F1_converges / check_3F2_converges: double -> int cast overflow

`stan/math/prim/err/check_2F1_converges.hpp:50` and `check_3F2_converges.hpp:58,66` assign
`num_terms = floor(fabs(value_of_rec(a)))` (an `int`) for a non-positive-integer `a` without a range check. Large or
infinite arguments (`1e300`, `inf`) trigger UBSan `float-cast-overflow` ("... is outside the range of representable values of type 'int'").
The same overflow appears in Eigen's `cast_impl<double,int>` when reached through the same call chain.

Reproducer: `hypergeometric_3F2_float_cast_overflow.bin` (run with `/mayhem/fuzz_special-standalone`).
Impact: undefined behavior on a legal input (the result is a garbage int, so the convergence check can be wrong).
Fix: check `std::isfinite` and the `int` range before casting, or keep the value as `double`.
