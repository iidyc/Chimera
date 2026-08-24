# Chimera

Chimera is a single-GPU multi-vector retrieval system. It builds a compressed
index from document-token embeddings and retrieves documents with late
interaction scoring.

The repository provides:

- a C++/CUDA library;
- command-line tools for index construction and evaluation;
- a Python binding exposing `build`, `search`, `save`, and `load`.

## Requirements

- 64-bit Linux
- an NVIDIA GPU and a compatible NVIDIA driver
- Git
- Conda

Python, the CUDA compiler, cuVS, RMM, CMake, Ninja, and pybind11 are installed
inside the Conda environment by the setup script.

## Build

Run the two setup scripts from the repository root:

```bash
./setup/setup_chimera_env.sh
./setup/build_chimera.sh
```

The first script creates a Conda environment named `chimera`. The second
script builds the project in `build/` and produces:

- `build/gpu_build`
- `build/gpu_search`
- `build/chimera*.so`

To select a GPU, set `CUDA_VISIBLE_DEVICES` when running an executable or
Python program.

## Input data

The CLI consumes three raw binary files and one TSV file. 

The layouts below use these symbols:

| Symbol | Meaning |
| --- | --- |
| `N` | total number of document-token embeddings |
| `D` | embedding dimension (currently 128) |
| `M` | number of documents |
| `Q` | number of queries |
| `L` | tokens per query (currently 32) |

Currently `D` is set to 128 and `L` is set to 32, compatible with ColBERTv2 model's default. To use different values, edit `config.cuh` and rebuild Chimera with
`./setup/build_chimera.sh`. `PADDED_DIM(D)` must remain a positive multiple of 64,
and `Q_DOCLEN(L)` should remain a positive multiple of 16 for the current GPU and
AVX-512 scoring paths. 

### Document embeddings

`embeddings.bin` represents a `float32[N, D]` matrix:

| Byte offset | Size | Value |
| ---: | ---: | --- |
| `0` | `4` | `N` as `int32` |
| `4` | `4` | `D` as `int32` |
| `8` | `4 * N * D` | flattened embeddings as `float32` |

Embedding component `j` of token `i` is stored at flattened position
`i * D + j`. Tokens belonging to one document must occupy consecutive rows;
`doclens.bin` defines where each document starts and ends.

### Document lengths

`doclens.bin` represents an `int32[M]` vector:

| Byte offset | Size | Value |
| ---: | ---: | --- |
| `0` | `4` | `M` as `int32` |
| `4` | `4 * M` | `M` positive document lengths as `int32` |

For example, if `doc_lens = [3, 2, 4]`, the rows in `embeddings.bin` map to
documents as follows:

| Document ID | Embedding rows | Token count |
| ---: | --- | ---: |
| `0` | `[0, 3)` | `3` |
| `1` | `[3, 5)` | `2` |
| `2` | `[5, 9)` | `4` |

Thus document IDs are the zero-based positions in the document-length vector,
and the required invariant is:

```text
N = doc_lens[0] + doc_lens[1] + ... + doc_lens[M - 1]
```

### Queries

`queries.bin` represents a `float32[Q, L, D]` tensor:

| Byte offset | Size | Value |
| ---: | ---: | --- |
| `0` | `4` | `Q` as `int32` |
| `4` | `4` | `L` as `int32` |
| `8` | `4` | `D` as `int32` |
| `12` | `4 * Q * L * D` | flattened query embeddings as `float32` |

Query `q`, token `t`, component `j` is stored at flattened position
`(q * L + t) * D + j`. In other words, all 32 token vectors for query 0 come
first, followed by all 32 token vectors for query 1, and so on. With the
default build, the kernels require `L = Q_DOCLEN = 32`; query dimension `D`
must equal the indexed document dimension.

### Ground truth

The evaluation CLI expects a text file with one query-document pair per line.
Each line has at least three tab-separated columns:

```text
query_id<TAB>document_id<TAB>rank
```

For example:

```text
0<TAB>42<TAB>1
0<TAB>7<TAB>2
1<TAB>105<TAB>1
```

`<TAB>` above denotes an actual tab character. Query and document IDs are
zero-based; ranks are one-based and must be between 1 and 1000. Extra columns
are ignored.

### Writing the binary files with NumPy

The following helpers generate files with exactly the layout expected by the
CLI:

```python
import numpy as np


def write_embeddings(path, embeddings):
    embeddings = np.ascontiguousarray(embeddings, dtype="<f4")
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [N, D]")
    with open(path, "wb") as output:
        np.asarray(embeddings.shape, dtype="<i4").tofile(output)
        embeddings.tofile(output)


def write_doc_lens(path, doc_lens):
    doc_lens = np.ascontiguousarray(doc_lens, dtype="<i4")
    if doc_lens.ndim != 1:
        raise ValueError("doc_lens must have shape [M]")
    with open(path, "wb") as output:
        np.asarray([doc_lens.size], dtype="<i4").tofile(output)
        doc_lens.tofile(output)


def write_queries(path, queries):
    queries = np.ascontiguousarray(queries, dtype="<f4")
    if queries.ndim != 3:
        raise ValueError("queries must have shape [Q, 32, D]")
    with open(path, "wb") as output:
        np.asarray(queries.shape, dtype="<i4").tofile(output)
        queries.tofile(output)
```

## Command-line usage

### Build an index

```bash
./build/gpu_build \
  --index indexes/example \
  --doclens data/doclens.bin \
  --data data/embeddings.bin \
  --n-clusters 4096 \
  --ex-bits 4
```

`--n-clusters` controls the number of clusters and must not exceed the
number of token embeddings. We recommend setting it sufficiently large (around N/200-N/100) `--ex-bits` controls the residual quantization
width and must be between 1 and 7.

Run `./build/gpu_build --help` for the complete argument list.

### Search and evaluate

```bash
./build/gpu_search \
  --query data/queries.bin \
  --gt data/groundtruth.tsv \
  --index indexes/example \
  --k 100 \
  --nprobe 128 \
  --k-refine 3000 \
  --k-full-bit 300 \
  --cagra-itopk-size 150 \
  --num-chunks 5
```

The program reports per-query latency statistics and recall. The tuning
parameters have the following roles:

- `nprobe`: clusters probed for each query token;
- `k-refine`: candidate documents retained for refinement;
- `k-full-bit`: refined candidates rescored with full-bit codes;
- `cagra-itopk-size`: intermediate CAGRA top-k size;
- `num-chunks`: chunks used by collaborative document scoring.

`k-full-bit` must not exceed `k-refine`, and `nprobe` must not exceed the
number of clusters in the index. It's sufficient to set `cagra-itopk-size` slightly larger than `nprobe`. Run `./build/gpu_search --help` for defaults
and the complete argument list.

## C++ API

Include `chimera/chimera_index.cuh` and link the application against the
`chimera_core` CMake target:

```cmake
add_subdirectory(path/to/Chimera)
target_link_libraries(my_application PRIVATE chimera_core)
```

The index accepts embeddings that have already been loaded into memory:

```cpp
#include "chimera/chimera_index.cuh"

#include <cstddef>
#include <vector>

std::vector<float> embeddings = load_document_embeddings();
std::vector<int> doc_lens = load_document_lengths();
const std::size_t dimension = 128;

Chimera::chimera_index index;
index.build(
    embeddings,
    dimension,
    doc_lens,
    /*n_clusters=*/4096,
    /*ex_bits=*/4);
index.save("indexes/example");

std::vector<float> query = load_query_embeddings();  // [32, dimension]
std::vector<std::size_t> document_ids = index.search(query, 100);
```

An existing index can be loaded with explicit search parameters:

```cpp
Chimera::SearchOptions options;
options.nprobe = 128;
options.k_refine = 3000;
options.k_full_bit = 300;
options.cagra_itopk_size = 150;
options.num_chunks = 4;

Chimera::chimera_index index;
index.load("indexes/example", options);
```

## Python usage

Add the build directory to `PYTHONPATH` and run Python from the Chimera Conda
environment:

```bash
export PYTHONPATH="$PWD/build${PYTHONPATH:+:$PYTHONPATH}"
conda activate chimera
```

Build, search, and save an index:

```python
import numpy as np
from chimera import ChimeraIndex

# Documents are represented by contiguous token embeddings.
embeddings = np.ascontiguousarray(
    np.load("embeddings.npy"), dtype=np.float32
)  # [num_tokens, dimension]
doc_lens = np.load("doclens.npy").astype(np.int32).tolist()

index = ChimeraIndex.build(
    embeddings,
    doc_lens,
    n_clusters=4096,
    ex_bits=4,
)

queries = np.ascontiguousarray(
    np.load("queries.npy"), dtype=np.float32
)  # [num_queries, 32, dimension]
document_ids = index.search(queries, k=100)
index.save("indexes/example")
```

Load an existing index and search one query:

```python
import numpy as np
from chimera import ChimeraIndex

index = ChimeraIndex.load(
    "indexes/example",
    nprobe=128,
    k_refine=3000,
    k_full_bit=300,
    cagra_itopk_size=150,
    num_chunks=4,
)

query = np.ascontiguousarray(
    np.load("query.npy"), dtype=np.float32
)  # [32, dimension]
document_ids = index.search(query, k=100)
```

Inputs passed through Python must be C-contiguous `float32` NumPy arrays.
For a single query, `search` returns `list[int]`; for a query batch, it returns
`list[list[int]]`.
