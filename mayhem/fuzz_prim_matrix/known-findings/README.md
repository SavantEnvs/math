# to_matrix(std::vector<double>) on an empty vector

`stan::math::to_matrix(const std::vector<double>&)` takes `&v[0]` of the vector, so an empty vector binds a
reference to a null pointer. UBSan (`-fsanitize=undefined`, `null`) halts at
`stl_vector.h:1131: reference binding to null pointer of type 'double'`.

Reproducer: `to_matrix_empty_vector.bin` (run with `/mayhem/fuzz_prim_matrix-standalone`).
Impact: undefined behavior on a legal input (no crash in practice at -O0 or -O2).
Fix: guard the empty case (`v.empty() ? Matrix(0, 0) : Map(v.data(), ...)`) or use `v.data()`.
