#!/usr/bin/env bash
# Builds, in /mayhem: the Stan Math fuzz harnesses (+ -standalone repro binaries) with sanitizers, and
# the gtest unit tests listed in mayhem/test.sh with the project's normal flags. Header-only library, so
# everything is resolved from the vendored lib/ tree: no network.
set -euo pipefail

[ -n "${SOURCE_DATE_EPOCH:-}" ] || unset SOURCE_DATE_EPOCH
: "${SANITIZER_FLAGS=-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer}"
: "${DEBUG_FLAGS:=-g -gdwarf-3}"
: "${CC:=clang}" ; : "${CXX:=clang++}" ; : "${LIB_FUZZING_ENGINE:=-fsanitize=fuzzer}"
: "${MAYHEM_JOBS:=$(nproc)}"
: "${COVERAGE_FLAGS=}"
export SANITIZER_FLAGS DEBUG_FLAGS CC CXX LIB_FUZZING_ENGINE MAYHEM_JOBS COVERAGE_FLAGS

cd "$SRC"
# A sanitized Stan translation unit peaks around 4-5 GB of RAM, so cap parallelism by the memory limit.
mem_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo)
for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
  [ -r "$f" ] && lim=$(cat "$f") && [[ "$lim" =~ ^[0-9]+$ ]] && [ $((lim / 1024)) -lt "$mem_kb" ] && mem_kb=$((lim / 1024))
done
mem_jobs=$(( mem_kb / 5000000 )); [ "$mem_jobs" -ge 1 ] || mem_jobs=1
[ "$MAYHEM_JOBS" -le "$mem_jobs" ] || MAYHEM_JOBS=$mem_jobs
OUT=/mayhem
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# 1) Test suite, normal flags (the project's own makefile). Built first, in-tree.
TESTS="prim/fun/log_sum_exp prim/fun/cholesky_decompose prim/fun/matrix_exp prim/prob/normal rev/fun/log_sum_exp rev/fun/log_softmax"
targets=""
for t in $TESTS; do targets="$targets test/unit/math/${t}_test"; done
# shellcheck disable=SC2086
make -j"$MAYHEM_JOBS" CC="$CC" CXX="$CXX" O=0 CPPFLAGS="$COVERAGE_FLAGS" LDFLAGS="$COVERAGE_FLAGS" $targets

# 2) Fuzz harnesses. Boost asserts stay off (as in Stan's own builds), Eigen's stay on.
FLAGS="-std=c++17 -O1 -I$SRC -isystem $SRC/lib/eigen_5.0.1 -isystem $SRC/lib/boost_1.87.0 \
 -isystem $SRC/lib/sundials_6.1.1/include -isystem $SRC/lib/tbb_2020.3/include \
 -DBOOST_DISABLE_ASSERTS -D_REENTRANT -Wno-deprecated-declarations -Wno-deprecated-enum-enum-conversion"
# Stan's rev code needs the TBB runtime and the SUNDIALS archives that step 1's makefile built (unsanitized).
SUN=$SRC/lib/sundials_6.1.1/lib
LINK="$SUN/libsundials_arkode.a $SUN/libsundials_cvodes.a $SUN/libsundials_idas.a $SUN/libsundials_kinsol.a $SUN/libsundials_nvecserial.a \
 $SRC/lib/tbb/libtbb.so.2 -Wl,-rpath,$SRC/lib/tbb"

$CC $SANITIZER_FLAGS $DEBUG_FLAGS -c "$STANDALONE_FUZZ_MAIN" -o "$WORK/standalone_main.o"
$CXX $SANITIZER_FLAGS -c mayhem/lsan_off.cc -o "$WORK/lsan_off.o"

BOOSTFUZZ=lib/boost_1.87.0/libs/algorithm/minmax/fuzzing
# name:source[:extra flags]. The upstream Boost harnesses use assert(), so NDEBUG/BOOST_DISABLE_ASSERTS stay off for them.
JOBS=(
  "fuzz_prim_matrix:mayhem/fuzz_prim_matrix.cpp"
  "fuzz_rev_matrix:mayhem/fuzz_rev_matrix.cpp"
  "fuzz_distributions:mayhem/fuzz_distributions.cpp"
  "fuzz_constraints:mayhem/fuzz_constraints.cpp"
  "fuzz_special:mayhem/fuzz_special.cpp"
  "fuzz_functors:mayhem/fuzz_functors.cpp"
  "minmax_element:$BOOSTFUZZ/minmax_element.fuzz.cpp"
  "minmax_element_variants:$BOOSTFUZZ/minmax_element_variants.fuzz.cpp"
)
# The Boost harnesses use uint8_t without including <cstdint>, and rely on assert() (no NDEBUG).
BOOST_ONLY="-include cstdint -UBOOST_DISABLE_ASSERTS"

build_one() {
  local name=${1%%:*} src=${1#*:} mode=$2 out extra opts="" link
  case $name in minmax_*) opts=$BOOST_ONLY; link="" ;; *) link=$LINK ;; esac
  if [ "$mode" = fuzz ]; then out=$OUT/$name; extra=$LIB_FUZZING_ENGINE
  else out=$OUT/$name-standalone; extra="$WORK/standalone_main.o"; fi
  # shellcheck disable=SC2086
  $CXX $SANITIZER_FLAGS $DEBUG_FLAGS $FLAGS $opts "$SRC/$src" $extra "$WORK/lsan_off.o" $link -o "$out"
}
export -f build_one
export FLAGS OUT WORK LINK BOOST_ONLY

for j in "${JOBS[@]}"; do printf '%s fuzz\n%s standalone\n' "$j" "$j"; done |
  xargs -P "$MAYHEM_JOBS" -L1 bash -c 'build_one "$0" "$1"'
