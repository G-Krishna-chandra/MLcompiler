// mlc_dump_kv_cache: prefill a prompt on the CPU backend, dumping per-step
// activations (any registered tap names) plus the final K/V cache state for
// block 0. Produces files in --out-dir matching the parity-harness sanitization
// scheme: <sanitized_name>.f32.bin (final cache) and step<N>_<sanitized_name>.f32.bin
// (per-step tap snapshots).
//
// Usage:
//   mlc_dump_kv_cache --model PATH --prompt "..." --out-dir DIR
//                     [--tap blk.0.attn_k.out] [--tap blk.0.attn_v.out] ...
//
// This is a one-off diagnostic for PART G of the attention investigation —
// it captures enough state to reconstruct the expected K/V cache contents
// independently from mlc's internal cache write logic.

#include "frontends/gguf_loader.hpp"
#include "runtime/execution_context.hpp"
#include "runtime/execution_executor.hpp"
#include "runtime/execution_plan_builder.hpp"
#include "runtime/kernel_registry.hpp"
#include "runtime/operator_backend.hpp"
#include "runtime/session.hpp"
#include "runtime/tokenizer.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::string sanitize(const std::string& s) {
    std::string out = s;
    for (auto& c : out) if (c == '.' || c == '/') c = '_';
    return out;
}

bool writeFloats(const std::string& path, const float* data, size_t n) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n * sizeof(float)));
    return out.good();
}

void usage() {
    std::cerr << "Usage: mlc_dump_kv_cache --model PATH --prompt \"...\" --out-dir DIR "
                 "[--tap NAME ...]\n";
}

} // namespace

int main(int argc, char** argv) {
    using namespace mlc::runtime;

    std::string model_path, prompt, out_dir;
    std::vector<std::string> taps;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--prompt" && i + 1 < argc) prompt = argv[++i];
        else if (a == "--out-dir" && i + 1 < argc) out_dir = argv[++i];
        else if (a == "--tap" && i + 1 < argc) taps.emplace_back(argv[++i]);
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { usage(); return 1; }
    }
    if (model_path.empty() || prompt.empty() || out_dir.empty()) { usage(); return 1; }

    try {
        Session session(model_path);

        Tokenizer tokenizer(session.loader());
        if (!tokenizer.valid()) {
            std::cerr << "Tokenizer init failed for " << model_path << "\n";
            return 1;
        }
        TokenizerConfig tcfg;
        tcfg.add_bos = true;
        tcfg.add_eos = false;
        auto tokens = tokenizer.encode(prompt, tcfg);
        if (tokens.empty()) {
            std::cerr << "Tokenizer produced no tokens\n";
            return 1;
        }
        std::cerr << "Tokenized: " << tokens.size() << " tokens\n";

        // Force CPU for clean parity reasoning.
        const bool prior = KernelDescriptorRegistry::forceCpu();
        KernelDescriptorRegistry::setForceCpu(true);

        auto graph = ExecutionPlanBuilder::BuildFromLoader(session.loader());
        ExecutionContext context(session, &graph);
        ExecutionExecutor executor(graph, &BackendRegistry::Default(), &context);

        for (const auto& t : taps) context.registerTap(t);

        for (size_t step = 0; step < tokens.size(); ++step) {
            uint64_t tok = tokens[step];
            context.setToken(tok);
            context.setSequencePosition(step);
            for (const auto& tname : {std::string("tokens"), std::string("token_ids")}) {
                if (graph.tensors().count(tname)) {
                    context.setTensor(tname, {static_cast<float>(tok)});
                }
            }
            auto res = executor.run();
            if (!res.success) {
                std::cerr << "Prefill failed at step " << step << "\n";
                KernelDescriptorRegistry::setForceCpu(prior);
                return 1;
            }
            // Dump current tap snapshots with step prefix.
            for (const auto& [name, data] : context.tapData()) {
                std::string path = out_dir + "/step" + std::to_string(step) + "_"
                                   + sanitize(name) + ".f32.bin";
                if (!writeFloats(path, data.data(), data.size())) {
                    std::cerr << "Failed to write " << path << "\n";
                }
            }
            std::cerr << "[step " << step << "] token=" << tok << " taps_dumped="
                      << context.tapData().size() << "\n";
        }

        // Dump K/V cache for block 0.
        for (const auto& cache_name : {std::string("kv_cache_k.0"), std::string("kv_cache_v.0")}) {
            const TensorStorage* storage = context.tensorStorage(cache_name);
            if (!storage) {
                std::cerr << "Cache tensor '" << cache_name << "' not found in context\n";
                continue;
            }
            if (storage->float_data.empty()) {
                std::cerr << "Cache '" << cache_name << "' has no float_data (dtype="
                          << storage->dtype << ")\n";
                continue;
            }
            std::string path = out_dir + "/" + sanitize(cache_name) + ".f32.bin";
            if (!writeFloats(path, storage->float_data.data(), storage->float_data.size())) {
                std::cerr << "Failed to write " << path << "\n";
                continue;
            }
            std::cerr << "[cache] " << cache_name << " -> " << path
                      << " (" << storage->float_data.size() << " floats)\n";
        }

        // Also report the rotary/freq config so the Python comparison can mirror it.
        if (graph.modelConfig().rotary_dim > 0) {
            std::cerr << "[rope] rotary_dim=" << graph.modelConfig().rotary_dim
                      << " freq_base=" << graph.modelConfig().rope_freq_base
                      << " freq_scale=" << graph.modelConfig().rope_freq_scale << "\n";
        }

        KernelDescriptorRegistry::setForceCpu(prior);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
