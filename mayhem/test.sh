#!/usr/bin/env bash
# Runs the Stan Math gtest unit tests that mayhem/build.sh compiled (normal flags) and reports CTRF counts.
set -uo pipefail
[ -n "${SOURCE_DATE_EPOCH:-}" ] || unset SOURCE_DATE_EPOCH
cd "$SRC"

TESTS="prim/fun/log_sum_exp prim/fun/cholesky_decompose prim/fun/matrix_exp prim/prob/normal rev/fun/log_sum_exp rev/fun/log_softmax"

emit_ctrf() {
  local tool="$1" passed="$2" failed="$3" skipped="${4:-0}" pending="${5:-0}" other="${6:-0}"
  local tests=$(( passed + failed + skipped + pending + other ))
  cat > "${CTRF_REPORT:-$SRC/ctrf-report.json}" <<JSON
{
  "results": {
    "tool": { "name": "$tool" },
    "summary": {
      "tests": $tests,
      "passed": $passed,
      "failed": $failed,
      "pending": $pending,
      "skipped": $skipped,
      "other": $other
    }
  }
}
JSON
  printf 'CTRF {"results":{"tool":{"name":"%s"},"summary":{"tests":%d,"passed":%d,"failed":%d,"pending":%d,"skipped":%d,"other":%d}}}\n' \
    "$tool" "$tests" "$passed" "$failed" "$pending" "$skipped" "$other"
  [ "$failed" -eq 0 ]
}

passed=0 failed=0 skipped=0
for t in $TESTS; do
  bin=test/unit/math/${t}_test
  if [ ! -x "$bin" ]; then echo "missing test binary $bin (mayhem/build.sh should have built it)" >&2; failed=$((failed + 1)); continue; fi
  out=$("$bin" 2>&1); rc=$?
  echo "$out" | tail -n 6
  # gtest prints "[  PASSED  ] N test(s)." / "[  FAILED  ] N test(s), listed below:" / "[  SKIPPED ] N test(s)"
  p=$(echo "$out" | sed -nE 's/^\[  PASSED  \] ([0-9]+) tests?\..*/\1/p' | tail -n1)
  f=$(echo "$out" | sed -nE 's/^\[  FAILED  \] ([0-9]+) tests?, listed below.*/\1/p' | tail -n1)
  s=$(echo "$out" | sed -nE 's/^\[  SKIPPED \] ([0-9]+) tests?, listed below.*/\1/p' | tail -n1)
  p=${p:-0} f=${f:-0} s=${s:-0}
  if [ "$rc" -ne 0 ] && [ "$f" -eq 0 ]; then f=1; fi
  if [ "$p" -eq 0 ] && [ "$f" -eq 0 ]; then echo "$bin ran no passing tests" >&2; f=1; fi
  passed=$((passed + p)); failed=$((failed + f)); skipped=$((skipped + s))
done

emit_ctrf "gtest" "$passed" "$failed" "$skipped"
