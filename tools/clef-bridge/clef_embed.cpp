// clef_embed: feed exact token ids to a GGUF model and dump per-token hidden
// states (the joint schema head's input). Private research helper.
//
//   single:  clef_embed <model.gguf> <tokens.txt> <out.bin>  [n_ctx] [n_gpu_layers]
//   batch:   clef_embed <model.gguf> <tokens_dir> <out_dir>  [n_ctx] [n_gpu_layers]
//
// Batch mode loads the model once and decodes every *.tok in the directory with
// a fresh context per record (so no cross-record state leaks), writing
// <stem>.bin next to it in the output dir.  Per-record timing goes to stderr as
// "REC <stem> <n_tokens> <ms>", followed by a "TOTAL ..." line.  Private
// research helper; no GPU layers by default (CPU keeps it off the display card).
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <filesystem>
namespace fs = std::filesystem;

static bool read_tokens(const std::string & path, std::vector<llama_token> & tokens) {
    FILE * f = fopen(path.c_str(), "r");
    if (f == nullptr) {
        return false;
    }
    int token = 0;
    while (fscanf(f, "%d", &token) == 1) {
        tokens.push_back(token);
    }
    fclose(f);
    return !tokens.empty();
}

static bool decode_one(llama_model * model, const std::string & tok_path,
                       const std::string & out_path, int n_ctx, bool quiet,
                       double * ms_out, long long * ntok_out) {
    std::vector<llama_token> tokens;
    if (!read_tokens(tok_path, tokens)) {
        fprintf(stderr, "clef_embed: failed to read tokens from %s\n", tok_path.c_str());
        return false;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = n_ctx;
    cparams.n_ubatch = n_ctx;
    cparams.embeddings = true;
    cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "clef_embed: failed to create context\n");
        return false;
    }

    // an adapter embedded in the model file (adapter.embedded=true) is attached
    // by this call at scale 1.0; without it a raw-API context ignores it
    llama_set_adapters_lora(ctx, nullptr, 0, nullptr);

    const auto t0 = std::chrono::steady_clock::now();
    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "clef_embed: decode failed for %s\n", tok_path.c_str());
        llama_free(ctx);
        return false;
    }
    const int n_embd = llama_model_n_embd(model);
    FILE * out = fopen(out_path.c_str(), "wb");
    if (out == nullptr) {
        fprintf(stderr, "clef_embed: failed to open %s\n", out_path.c_str());
        llama_free(ctx);
        return false;
    }
    for (size_t i = 0; i < tokens.size(); i++) {
        float * e = llama_get_embeddings_ith(ctx, (int32_t) i);
        if (e == nullptr) {
            fprintf(stderr, "clef_embed: no embedding for token %zu\n", i);
            fclose(out);
            llama_free(ctx);
            return false;
        }
        fwrite(e, sizeof(float), (size_t) n_embd, out);
    }
    fclose(out);
    llama_free(ctx);

    const auto t1 = std::chrono::steady_clock::now();
    if (ms_out) {
        *ms_out = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    if (ntok_out) {
        *ntok_out = (long long) tokens.size();
    }
    if (!quiet) {
        fprintf(stderr, "wrote %s: %zu tokens x %d floats\n", out_path.c_str(), tokens.size(), n_embd);
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr,
                "usage: clef_embed <model.gguf> <tokens.txt|tokens_dir> <out.bin|out_dir> [n_ctx] [n_gpu_layers]\n");
        return 2;
    }
    const bool batch = fs::is_directory(argv[2]);

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = (argc > 5) ? atoi(argv[5]) : 0;  // 0 = CPU
    const auto tl0 = std::chrono::steady_clock::now();
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (model == nullptr) {
        fprintf(stderr, "clef_embed: failed to load %s\n", argv[1]);
        return 1;
    }
    const double load_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - tl0).count();

    const int n_ctx = argc > 4 ? atoi(argv[4]) : 4096;
    int rc = 0;

    if (!batch) {
        double ms = 0.0;
        long long ntok = 0;
        if (!decode_one(model, argv[2], argv[3], n_ctx, false, &ms, &ntok)) {
            rc = 1;
        } else {
            fprintf(stderr, "REC %s %lld %.1f\n", argv[3], ntok, ms);
        }
    } else {
        fs::path out_dir = argv[3];
        if (!fs::is_directory(out_dir)) {
            fs::create_directories(out_dir);
        }
        std::vector<fs::path> toks;
        for (const auto & e : fs::directory_iterator(argv[2])) {
            if (e.path().extension() == ".tok") {
                toks.push_back(e.path());
            }
        }
        std::sort(toks.begin(), toks.end());
        int n = 0;
        double total = 0.0;
        for (const auto & p : toks) {
            const std::string stem = p.stem().string();
            const fs::path out = out_dir / (stem + ".bin");
            double ms = 0.0;
            long long ntok = 0;
            if (!decode_one(model, p.string(), out.string(), n_ctx, true, &ms, &ntok)) {
                rc = 1;
                break;
            }
            fprintf(stderr, "REC %s %lld %.1f\n", stem.c_str(), ntok, ms);
            total += ms;
            n++;
        }
        fprintf(stderr, "TOTAL %d records %.1f ms (model load %.1f ms, n_ctx %d)\n",
                n, total, load_ms, n_ctx);
        if (n == 0) {
            fprintf(stderr, "clef_embed: no *.tok files under %s\n", argv[2]);
            rc = 1;
        }
    }

    llama_model_free(model);
    llama_backend_free();
    return rc;
}
