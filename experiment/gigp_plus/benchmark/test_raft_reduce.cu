#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "include/alg/igp_gpu_pp/RaftReductionCompat.hpp"

void check(cudaError_t code) {
  if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}

template <bool AlongRows>
void test(uint32_t columns, uint32_t rows) {
  std::vector<float> input(columns * rows);
  for (size_t i = 0; i < input.size(); ++i) input[i] = (int(i % 17) - 8) * 0.25f;
  std::vector<float> expected(AlongRows ? rows : columns, 0.0f);
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t c = 0; c < columns; ++c)
      expected[AlongRows ? r : c] += input[r * columns + c];
  float *data, *output;
  check(cudaMalloc(&data, input.size() * sizeof(float)));
  check(cudaMalloc(&output, expected.size() * sizeof(float)));
  check(cudaMemcpy(data, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
  VectorSetSearch::reduceScoresCompat<true, AlongRows>(output, data, columns, rows, nullptr);
  check(cudaDeviceSynchronize());
  std::vector<float> actual(expected.size());
  check(cudaMemcpy(actual.data(), output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < actual.size(); ++i)
    if (std::abs(actual[i] - expected[i]) > 1e-5f) throw std::runtime_error("Reduction differs from CPU reference");
  check(cudaFree(data));
  check(cudaFree(output));
}

int main() {
  test<true>(3, 2);
  test<false>(3, 2);
  test<true>(32, 19);
  test<false>(32, 19);
  std::cout << "RAFT reduction compatibility: four GPU/CPU reference checks passed\n";
}
