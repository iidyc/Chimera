#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>
#include <cnpy.h>
#include "include/struct/IVFIndex.hpp"

int main(int argc, char** argv) {
  if (argc != 5) return 2;
  auto codes = cnpy::npy_load(argv[1]);
  auto lengths = cnpy::npy_load(argv[2]);
  assert(codes.word_size == 4 && lengths.word_size == 4);
  auto* c = codes.data<uint32_t>();
  auto* l = lengths.data<uint32_t>();
  VectorSetSearch::Method::IVFIndex index(
      std::vector<uint32_t>(c, c + codes.num_vals),
      std::vector<uint32_t>(l, l + lengths.num_vals),
      codes.num_vals, lengths.num_vals, std::stoul(argv[3]));
  index.build();
  index.saveIVFIndex(argv[4]);
}
