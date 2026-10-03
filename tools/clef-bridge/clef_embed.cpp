// clef_embed: feed exact token ids to a GGUF model and dump per-token hidden
// states (the joint schema head's input). Private research helper.
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: clef_embed <model.gguf> <tokens.txt> <out.bin> [n_ctx]\n");
        return 2;
    }
    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = (argc > 5) ? atoi(argv[5]) : 0;  // 0 = CPU
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (model == nullptr) {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = argc > 4 ? atoi(argv[4]) : 4096;
    cparams.n_batch = cparams.n_ctx;
    cparams.n_ubatch = cparams.n_ctx;
    cparams.embeddings = true;
    cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }
    std::vector<llama_token> tokens;
    FILE * f = fopen(argv[2], "r");
    if (f == nullptr) {
        fprintf(stderr, "failed to open %s\n", argv[2]);
        return 1;
    }
    int token = 0;
    while (fscanf(f, "%d", &token) == 1) {
        tokens.push_back(token);
    }
    fclose(f);
    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }
    const int n_embd = llama_model_n_embd(model);
    FILE * out = fopen(argv[3], "wb");
    if (out == nullptr) {
        fprintf(stderr, "failed to open %s\n", argv[3]);
        return 1;
    }
    for (size_t i = 0; i < tokens.size(); i++) {
        float * e = llama_get_embeddings_ith(ctx, (int32_t) i);
        if (e == nullptr) {
            fprintf(stderr, "no embedding for token %zu\n", i);
            return 1;
        }
        fwrite(e, sizeof(float), (size_t) n_embd, out);
    }
    fclose(out);
    fprintf(stderr, "wrote %zu tokens x %d floats\n", tokens.size(), n_embd);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
