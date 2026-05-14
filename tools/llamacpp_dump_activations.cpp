// llamacpp_dump_activations: load a GGUF via llama.cpp, run a single prefill on
// the supplied prompt, and dump per-tensor activations at the boundaries that
// mlc's parity harness taps. Output format matches `mlc compare --vs-llamacpp`'s
// expectation: one binary file per tensor, raw little-endian float32, named
// <sanitized_mlc_name>.f32.bin in --out-dir.
//
// Usage:
//   llamacpp_dump_activations --model PATH --prompt "..." --out-dir DIR
//                             [--cache-type-k f16|f32] [--cache-type-v f16|f32]
//
// Cache types default to fp16 (matching llama.cpp's own
// llama_context_default_params() defaults and our fp16 KV cache path).
// Pass --cache-type-k f32 / --cache-type-v f32 to compare against an
// fp32-cache configuration explicitly.
//
// Tensor name mapping (llama.cpp graph node names → mlc canonical names):
//
//   inp_embd            -> hidden_state_0
//   kqv_out-N           -> blk.N.attn_output.out
//   ffn_inp-N           -> blk.N.residual_1
//   ffn_out-N           -> blk.N.ffn_down.out
//   l_out-N             -> blk.N.residual_2
//   result_norm         -> final_norm_out
//   result_output       -> logits
//
// llama.cpp uses these names in its Llama-family graph builder. If a llama.cpp
// version uses different names (we've seen "kq_out-N", "attn_out-N", "norm-N"
// in older builds), the unmatched tensor names are logged to stderr at the end
// of the run so the map can be updated.
//
// Tensor shape handling: llama.cpp prefills the entire prompt as one batch, so
// each per-layer activation has shape [hidden_dim, n_tokens]. mlc's parity
// harness, by contrast, prefills one token at a time and taps the LAST step.
// To make the comparison apples-to-apples, this tool extracts the LAST column
// (the activation for the last prompt token) before writing.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace {

std::string sanitizeForFilename(const std::string& name) {
    std::string out = name;
    for (auto& c : out) {
        if (c == '.' || c == '/') c = '_';
    }
    return out;
}

std::string mapLlamaToMlc(const std::string& llama_name) {
    // Exact-match prefixes (no layer suffix).
    if (llama_name == "inp_embd")      return "hidden_state_0";
    if (llama_name == "result_norm")   return "final_norm_out";
    if (llama_name == "result_output") return "logits";

    // Suffixed names: "<prefix>-<layer_int>".
    auto dash = llama_name.rfind('-');
    if (dash == std::string::npos) return "";
    std::string prefix = llama_name.substr(0, dash);
    std::string suffix = llama_name.substr(dash + 1);
    int layer = -1;
    try {
        size_t consumed = 0;
        layer = std::stoi(suffix, &consumed);
        if (consumed != suffix.size()) return "";
    } catch (...) {
        return "";
    }

    // Boundary tensors (default mlc tap set).
    if (prefix == "kqv_out") return "blk." + std::to_string(layer) + ".attn_output.out";
    if (prefix == "attn_out") return "blk." + std::to_string(layer) + ".attn_output.out";
    if (prefix == "ffn_inp") return "blk." + std::to_string(layer) + ".residual_1";
    if (prefix == "ffn_out") return "blk." + std::to_string(layer) + ".ffn_down.out";
    if (prefix == "l_out")   return "blk." + std::to_string(layer) + ".residual_2";
    // Intermediate sub-op tensors (for block-N drilldown). attn_norm/ffn_norm
    // emit the gamma-applied (post-multiply) result that matches mlc's
    // *.attn_norm.out / *.ffn_norm.out. Qcur/Kcur/Vcur are the matmul outputs
    // from llama.cpp; mlc's attn_q/k/v.out are also matmul outputs (RoPE is
    // applied inside the attention op in both backends, so these compare
    // pre-RoPE). __fattn__ is the merged-heads attention output.
    if (prefix == "attn_norm")   return "blk." + std::to_string(layer) + ".attn_norm.out";
    if (prefix == "Qcur")        return "blk." + std::to_string(layer) + ".attn_q.out";
    if (prefix == "Kcur")        return "blk." + std::to_string(layer) + ".attn_k.out";
    if (prefix == "Vcur")        return "blk." + std::to_string(layer) + ".attn_v.out";
    if (prefix == "__fattn__")   return "blk." + std::to_string(layer) + ".attention_mix";
    if (prefix == "ffn_norm")    return "blk." + std::to_string(layer) + ".ffn_norm.out";
    if (prefix == "ffn_gate")    return "blk." + std::to_string(layer) + ".ffn_gate.out";
    if (prefix == "ffn_up")      return "blk." + std::to_string(layer) + ".ffn_up.out";
    if (prefix == "ffn_swiglu")  return "blk." + std::to_string(layer) + ".ffn_mix";
    return "";
}

struct DumpCtx {
    std::string out_dir;
    std::set<std::string> dumped;          // mlc canonical names successfully written
    std::set<std::string> all_names_seen;  // every llama tensor name observed (for discovery)
    std::set<std::string> unmatched_names; // llama tensor names we didn't recognize
    std::vector<uint8_t> buf;              // scratch for non-host tensors
};

bool dumpCallback(ggml_tensor* t, bool ask, void* user_data) {
    auto* ctx = static_cast<DumpCtx*>(user_data);
    if (!t || !t->name) return false;
    std::string llama_name = t->name;
    if (ask) ctx->all_names_seen.insert(llama_name);

    std::string mlc_name = mapLlamaToMlc(llama_name);
    if (mlc_name.empty()) {
        if (ask) ctx->unmatched_names.insert(llama_name);
        return false;
    }

    if (ask) return true;
    if (ctx->dumped.count(mlc_name)) return true;

    if (ggml_is_quantized(t->type)) {
        std::fprintf(stderr, "[skip] %s is quantized (%s); expected activation, skipping\n",
                     llama_name.c_str(), ggml_type_name(t->type));
        return true;
    }

    // Get raw bytes (handle host vs device backends).
    bool is_host = ggml_backend_buffer_is_host(t->buffer);
    const size_t n_bytes = ggml_nbytes(t);
    const uint8_t* data = nullptr;
    if (is_host) {
        data = static_cast<const uint8_t*>(t->data);
    } else {
        ctx->buf.resize(n_bytes);
        ggml_backend_tensor_get(t, ctx->buf.data(), 0, n_bytes);
        data = ctx->buf.data();
    }

    // Convert to f32. Activations are usually F32 already; F16 also possible.
    const size_t n_elements = ggml_nelements(t);
    std::vector<float> f32(n_elements);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(f32.data(), data, n_elements * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t* src = reinterpret_cast<const ggml_fp16_t*>(data);
        for (size_t i = 0; i < n_elements; ++i) {
            f32[i] = ggml_fp16_to_fp32(src[i]);
        }
    } else {
        std::fprintf(stderr, "[skip] %s unexpected dtype %s\n",
                     llama_name.c_str(), ggml_type_name(t->type));
        return true;
    }

    // Find the last non-1 dimension and treat it as the batch (n_tokens). All
    // earlier dims contribute to per-token size. Handles both 2D activations
    // ([hidden, n_tokens]) and 3D attention outputs ([head_dim, n_heads,
    // n_tokens]); for the latter, we take the last token's full
    // head_dim*n_heads slice (= the merged-head hidden vector).
    int batch_dim = 0;
    for (int d = GGML_MAX_DIMS - 1; d >= 0; --d) {
        if (t->ne[d] > 1) { batch_dim = d; break; }
    }
    int64_t per_token = 1;
    for (int d = 0; d < batch_dim; ++d) per_token *= t->ne[d];
    int64_t batch = t->ne[batch_dim];
    int64_t last_col = batch - 1;
    if ((last_col + 1) * per_token > static_cast<int64_t>(n_elements)) {
        std::fprintf(stderr, "[skip] %s unexpected layout ne=[%lld,%lld,%lld,%lld] elements=%zu\n",
                     llama_name.c_str(), (long long)t->ne[0], (long long)t->ne[1],
                     (long long)t->ne[2], (long long)t->ne[3], n_elements);
        return true;
    }
    std::vector<float> last_token(static_cast<size_t>(per_token));
    std::memcpy(last_token.data(), f32.data() + last_col * per_token,
                static_cast<size_t>(per_token) * sizeof(float));

    // Write to disk.
    std::string path = ctx->out_dir + "/" + sanitizeForFilename(mlc_name) + ".f32.bin";
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::fprintf(stderr, "[err] failed to open %s\n", path.c_str());
        return true;
    }
    out.write(reinterpret_cast<const char*>(last_token.data()),
              static_cast<std::streamsize>(per_token) * sizeof(float));
    if (!out) {
        std::fprintf(stderr, "[err] failed to write %s\n", path.c_str());
        return true;
    }
    ctx->dumped.insert(mlc_name);
    std::fprintf(stderr, "[dump] %-30s -> %-32s (%lld floats; ne=[%lld,%lld,%lld,%lld] last_col=%lld)\n",
                 llama_name.c_str(), mlc_name.c_str(),
                 (long long)per_token,
                 (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3],
                 (long long)last_col);
    return true;
}

void usage() {
    std::cerr << "Usage: llamacpp_dump_activations --model PATH --prompt \"...\" --out-dir DIR\n"
                 "                                  [--cache-type-k f16|f32]\n"
                 "                                  [--cache-type-v f16|f32]\n"
                 "Cache types default to fp16 (llama.cpp's default; matches our fp16 KV path).\n";
}

ggml_type parseCacheType(const std::string& s, const char* flag) {
    if (s == "f16" || s == "fp16" || s == "F16") return GGML_TYPE_F16;
    if (s == "f32" || s == "fp32" || s == "F32") return GGML_TYPE_F32;
    std::cerr << flag << ": unknown cache type '" << s << "' (expected f16 or f32)\n";
    std::exit(1);
}

} // namespace

int main(int argc, char** argv) {
    std::string model_path, prompt, out_dir;
    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc)        model_path = argv[++i];
        else if (a == "--prompt" && i + 1 < argc)  prompt = argv[++i];
        else if (a == "--out-dir" && i + 1 < argc) out_dir = argv[++i];
        else if (a == "--cache-type-k" && i + 1 < argc) type_k = parseCacheType(argv[++i], "--cache-type-k");
        else if (a == "--cache-type-v" && i + 1 < argc) type_v = parseCacheType(argv[++i], "--cache-type-v");
        else if (a == "-h" || a == "--help")       { usage(); return 0; }
        else { usage(); return 1; }
    }
    if (model_path.empty() || prompt.empty() || out_dir.empty()) { usage(); return 1; }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap = true;
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        std::cerr << "Failed to load model: " << model_path << "\n";
        return 1;
    }

    DumpCtx cb_data;
    cb_data.out_dir = out_dir;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.type_k = type_k;
    cparams.type_v = type_v;
    cparams.cb_eval = dumpCallback;
    cparams.cb_eval_user_data = &cb_data;
    std::cerr << "[config] cache_type_k="
              << (type_k == GGML_TYPE_F16 ? "f16" : "f32")
              << " cache_type_v="
              << (type_v == GGML_TYPE_F16 ? "f16" : "f32") << "\n";

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::cerr << "Failed to init llama context\n";
        llama_model_free(model);
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    int32_t needed = llama_tokenize(vocab, prompt.c_str(),
                                    static_cast<int32_t>(prompt.size()),
                                    nullptr, 0, /*add_special=*/true, /*parse_special=*/false);
    if (needed < 0) needed = -needed;
    std::vector<llama_token> tokens(static_cast<size_t>(needed));
    int32_t written = llama_tokenize(vocab, prompt.c_str(),
                                     static_cast<int32_t>(prompt.size()),
                                     tokens.data(), needed,
                                     /*add_special=*/true, /*parse_special=*/false);
    if (written < 0) written = -written;
    tokens.resize(static_cast<size_t>(written));

    std::cerr << "Tokenized prompt (" << tokens.size() << " tokens):";
    for (auto t : tokens) std::cerr << " " << static_cast<int>(t);
    std::cerr << "\n";

    llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
    if (llama_decode(ctx, batch) != 0) {
        std::cerr << "llama_decode failed\n";
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    std::cerr << "\n[summary] dumped " << cb_data.dumped.size() << " tensors to " << out_dir << "\n";
    std::cerr << "[summary] " << cb_data.all_names_seen.size()
              << " distinct llama.cpp tensor names observed; "
              << cb_data.unmatched_names.size() << " unmapped.\n";
    // For block-0 sub-op discovery: print every observed name whose llama suffix
    // looks layer-zero (or layerless) so we can extend the map.
    if (std::getenv("LLAMA_DUMP_LIST_ALL")) {
        std::cerr << "\n[all observed llama.cpp tensor names]:\n";
        for (const auto& n : cb_data.all_names_seen) {
            std::cerr << "  " << n << "\n";
        }
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
