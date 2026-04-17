#include "arg_utils.hpp"
#include "build_gpu_index_fast.hpp"
#include "gpu_index_layout.hpp"
#include "io.hpp"

#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

namespace {

void copy_doclens_into_index_dir(
    const std::string& index_dir,
    const std::string& doclens_filename)
{
    std::filesystem::create_directories(index_dir);
    const auto dest_doclens_path = gpu_index_layout::doclens_path(index_dir);
    const auto source_path = std::filesystem::path(doclens_filename);
    const auto dest_path = std::filesystem::path(dest_doclens_path);
    if (std::filesystem::exists(source_path) &&
        std::filesystem::exists(dest_path) &&
        std::filesystem::equivalent(source_path, dest_path)) {
        return;
    }
    std::filesystem::copy_file(
        source_path,
        dest_path,
        std::filesystem::copy_options::overwrite_existing);
}

void print_input_help(const char* program) {
    std::cout
        << "Usage: " << program
        << " --index_dir <index_dir> --doclens <doclens> [--data <data> --n_clusters <n_clusters> | --source_index <index_dir> --clustered_only]\n\n"
        << "Arguments:\n"
        << "  --index_dir   Output index directory.\n"
        << "                The builder writes three files into this directory:\n"
        << "                ivf.bin, doc_1bit.bin, doc_4bit.bin, cluster_1bit.bin,\n"
        << "                index_metadata.json, and centroids.carga.\n"
        << "  --doclens     Input doclens file.\n"
        << "                Each value is the length of one document, i.e. the number of\n"
        << "                token embeddings belonging to that document.\n"
        << "                This is required to recover document embeddings from token embeddings.\n"
        << "                The file is copied into <index_dir>/doclens.bin.\n"
        << "  --data        Input token embedding file.\n"
        << "                This contains the token embeddings used to build the index.\n"
        << "  --source_index Existing split index directory to clone before generating\n"
        << "                cluster_1bit.bin only.\n"
        << "  --clustered_only Generate only cluster_1bit.bin from an existing index.\n"
        << "  --n_clusters  Number of randomly sampled centroids used to build the CAGRA graph.\n\n"
        << "Summary:\n"
        << "  Full build mode needs doclens, data, and n_clusters.\n"
        << "  Fast full build mode loads raw embeddings once and overlaps GPU assignment\n"
        << "  with CPU quantization before generating cluster_1bit.bin from doc_1bit.bin.\n"
        << "  Outputs are built in a staging directory and only moved into place on success.\n"
        << "  Fast sidecar mode needs doclens, source_index, and clustered_only.\n";
}

std::filesystem::path make_unique_peer_path(
    const std::filesystem::path& anchor_path,
    const std::string& suffix)
{
    const auto parent = anchor_path.parent_path().empty()
        ? std::filesystem::path(".")
        : anchor_path.parent_path();
    const auto stem = anchor_path.filename().string();

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<unsigned long long> dist;

    for (int attempt = 0; attempt < 128; ++attempt)
    {
        const auto candidate =
            parent / (stem + suffix + "-" + std::to_string(dist(gen)));
        std::error_code exists_ec;
        if (!std::filesystem::exists(candidate, exists_ec))
        {
            return candidate;
        }
    }

    throw std::runtime_error(
        "Failed to allocate a unique staging path next to " + anchor_path.string());
}

std::filesystem::path make_staging_index_dir(const std::string& final_index_dir)
{
    const auto final_path = std::filesystem::path(final_index_dir);
    const auto staging_path = make_unique_peer_path(final_path, ".tmp-gpu_build_fast");
    std::filesystem::create_directories(staging_path);
    return staging_path;
}

void remove_tree_if_exists(const std::filesystem::path& path)
{
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
}

void commit_staged_index_dir(
    const std::filesystem::path& staging_path,
    const std::filesystem::path& final_path)
{
    const bool final_exists = std::filesystem::exists(final_path);
    std::filesystem::path backup_path;
    if (final_exists)
    {
        backup_path = make_unique_peer_path(final_path, ".bak-gpu_build_fast");
        std::filesystem::rename(final_path, backup_path);
    }

    try
    {
        std::filesystem::rename(staging_path, final_path);
    }
    catch (...)
    {
        if (final_exists && !std::filesystem::exists(final_path) &&
            std::filesystem::exists(backup_path))
        {
            std::filesystem::rename(backup_path, final_path);
        }
        remove_tree_if_exists(staging_path);
        throw;
    }

    if (final_exists)
    {
        remove_tree_if_exists(backup_path);
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string index_dir;
    std::string data_filename;
    std::string doclens_filename;
    std::string source_index_dir;
    size_t n_clusters = 0;
    bool clustered_only = false;

    if (argc == 1) {
        print_input_help(argv[0]);
        return 1;
    }

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                print_input_help(argv[0]);
                return 0;
            }
            if (arg == "--index_dir") {
                index_dir = require_value(argc, argv, i, arg);
                continue;
            }
            if (arg == "--doclens") {
                doclens_filename = require_value(argc, argv, i, arg);
                continue;
            }
            if (arg == "--data") {
                data_filename = require_value(argc, argv, i, arg);
                continue;
            }
            if (arg == "--source_index") {
                source_index_dir = require_value(argc, argv, i, arg);
                continue;
            }
            if (arg == "--clustered_only") {
                clustered_only = true;
                continue;
            }
            if (arg == "--n_clusters") {
                n_clusters = std::stoull(require_value(argc, argv, i, arg));
                continue;
            }

            throw std::runtime_error("Unknown argument: " + arg);
        }
    } catch (const std::exception& e) {
        std::cerr << "Argument error: " << e.what() << "\n\n";
        print_input_help(argv[0]);
        return 1;
    }

    if (index_dir.empty() || doclens_filename.empty()) {
        std::cerr << "Missing required arguments.\n\n";
        print_input_help(argv[0]);
        return 1;
    }

    auto doc_lens = load_doclens(doclens_filename);
    std::filesystem::path staging_index_dir;

    try {
        staging_index_dir = make_staging_index_dir(index_dir);
        copy_doclens_into_index_dir(staging_index_dir.string(), doclens_filename);

        if (clustered_only) {
            if (source_index_dir.empty()) {
                std::cerr << "Missing required arguments for clustered-only mode.\n\n";
                print_input_help(argv[0]);
                remove_tree_if_exists(staging_index_dir);
                return 1;
            }
            build_clustered_stage1_sidecar_fast(
                doc_lens,
                source_index_dir,
                staging_index_dir.string());
            commit_staged_index_dir(staging_index_dir, index_dir);
            return 0;
        }

        if (data_filename.empty() || n_clusters == 0) {
            std::cerr << "Missing required arguments for full build mode.\n\n";
            print_input_help(argv[0]);
            remove_tree_if_exists(staging_index_dir);
            return 1;
        }

        size_t ex_bits = 3;
        auto data = load_data_mmap(data_filename);
        build_index_fast(data, n_clusters, ex_bits, doc_lens, staging_index_dir.string());
        commit_staged_index_dir(staging_index_dir, index_dir);
        return 0;
    } catch (const std::exception& e) {
        if (!staging_index_dir.empty()) {
            remove_tree_if_exists(staging_index_dir);
        }
        std::cerr << "Build failed: " << e.what() << std::endl;
        return 1;
    }
}
