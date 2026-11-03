#pragma once

#include <raft/linalg/reduce.cuh>

namespace VectorSetSearch {
// RAFT moved layout/direction flags from runtime arguments to template arguments.
template <bool RowMajor, bool AlongRows, typename T, typename Index>
void reduceScoresCompat(T* output, const T* input, Index columns, Index rows,
                        cudaStream_t stream) {
#ifdef GIGP_RAFT_TEMPLATE_REDUCE
  raft::linalg::reduce<RowMajor, AlongRows>(output, input, columns, rows, T{0},
      stream, false, raft::identity_op(), raft::add_op(), raft::identity_op());
#else
  raft::linalg::reduce(output, input, columns, rows, T{0}, RowMajor, AlongRows,
      stream, false, raft::identity_op(), raft::add_op(), raft::identity_op());
#endif
}
}
