#pragma once

// Chimera compile-time configuration
// These values must match the index file being loaded.
// Recompile if these values need to change.

#define PADDED_DIM 128   // Dimension after rotation/padding (must be multiple of 64)
#define Q_DOCLEN 32      // Query document length (number of query tokens)

// Derived constants (computed at compile-time)
#define CODE_BYTES (PADDED_DIM / 8)             // Binary code size: 16 bytes
#define NUM_U64 (PADDED_DIM / 64)               // Number of 64-bit blocks: 2
#define SMEM_QUERY_SIZE (PADDED_DIM * sizeof(float))  // Shared memory per query: 512 bytes

// LUT-based binary IP configuration
#define BITS_PER_CHUNK 4
#define LUT_SIZE (1 << BITS_PER_CHUNK)                        // 16
#define NUM_CHUNKS (PADDED_DIM / BITS_PER_CHUNK)              // 32
#define LUT_ENTRIES_PER_QUERY (NUM_CHUNKS * LUT_SIZE)         // 512
#define LUT_BYTES_PER_QUERY (LUT_ENTRIES_PER_QUERY * sizeof(float))  // 2048
#define LUT_TOTAL_FLOATS (Q_DOCLEN * LUT_ENTRIES_PER_QUERY)   // 16384
#define LUT_TOTAL_BYTES (LUT_TOTAL_FLOATS * sizeof(float))    // 65536 = 64 KB

// Candidate-refinement partial-score reduction.
#define PARTIAL_SCORE_WARPS_PER_BLOCK 8

// Validation
static_assert(PADDED_DIM % 64 == 0, "PADDED_DIM must be a multiple of 64");
static_assert(PADDED_DIM > 0, "PADDED_DIM must be positive");
static_assert(Q_DOCLEN > 0, "Q_DOCLEN must be positive");
