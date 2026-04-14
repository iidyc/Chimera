#pragma once

struct gpu_search_runtime_options {
    int nprobe = 128;
    int k_rank_cluster = 0;
    int k_rank_all_tokens = 300;
    int itopk_size = 150;
    int overlap_chunks = 5;
};
