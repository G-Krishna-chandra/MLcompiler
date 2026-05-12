#include "runtime/parity_harness.hpp"

#include "runtime/execution_context.hpp"
#include "runtime/execution_executor.hpp"
#include "runtime/execution_plan_builder.hpp"
#include "runtime/kernel_registry.hpp"
#include "runtime/operator_backend.hpp"
#include "runtime/session.hpp"
#include "runtime/tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace mlc {
namespace runtime {
namespace parity {

namespace {

// Default boundary patterns for Llama-style models. Substring match against
// execution-graph tensor names.
const std::vector<std::string>& defaultTapPatterns() {
    static const std::vector<std::string> patterns = {
        "hidden_state_0",
        "attn_output.out",
        "residual_1",
        "ffn_down.out",
        "residual_2",
        "final_norm_out",
        "logits",
    };
    return patterns;
}

std::vector<std::string> findTapTensors(const ExecutionGraph& graph,
                                        const std::vector<std::string>& patterns) {
    std::vector<std::string> result;
    for (const auto& [name, info] : graph.tensors()) {
        (void)info;
        for (const auto& p : patterns) {
            if (!p.empty() && name.find(p) != std::string::npos) {
                result.push_back(name);
                break;
            }
        }
    }
    // Sort by topological execution order so the output reads layer-by-layer.
    auto order = graph.topologicalOrder();
    std::unordered_map<std::string, size_t> node_pos;
    for (size_t i = 0; i < order.size(); ++i) node_pos[order[i]] = i;
    std::unordered_map<std::string, size_t> tensor_pos;
    for (const auto& node : graph.nodes()) {
        size_t pos = node_pos.count(node.name) ? node_pos[node.name]
                                               : std::numeric_limits<size_t>::max();
        for (const auto& out : node.outputs) {
            tensor_pos[out] = pos;
        }
    }
    std::sort(result.begin(), result.end(), [&](const std::string& a, const std::string& b) {
        size_t pa = tensor_pos.count(a) ? tensor_pos[a] : std::numeric_limits<size_t>::max();
        size_t pb = tensor_pos.count(b) ? tensor_pos[b] : std::numeric_limits<size_t>::max();
        if (pa != pb) return pa < pb;
        return a < b;
    });
    return result;
}

std::vector<uint64_t> tokenize(const Session& session,
                               const std::string& prompt,
                               bool wrap_chat) {
    Tokenizer tokenizer(session.loader());
    if (!tokenizer.valid()) {
        throw std::runtime_error("Tokenizer unavailable for the given GGUF");
    }
    std::string text = prompt;
    if (wrap_chat) {
        text = "[INST] <<SYS>>You are a helpful assistant.<</SYS>>\n" + text + " [/INST]";
    }
    TokenizerConfig cfg;
    cfg.add_bos = true;
    cfg.add_eos = false;
    return tokenizer.encode(text, cfg);
}

std::unordered_map<std::string, std::vector<float>>
runPrefillAndCaptureTaps(Session& session,
                         const std::vector<uint64_t>& tokens,
                         const std::vector<std::string>& tap_patterns,
                         std::vector<std::string>* tap_names_out,
                         std::string* note_out,
                         std::unordered_map<std::string, BackendKind>* dispatch_trace_out = nullptr) {
    auto graph = ExecutionPlanBuilder::BuildFromLoader(session.loader());
    auto taps = findTapTensors(graph, tap_patterns);
    if (tap_names_out) *tap_names_out = taps;

    ExecutionContext context(session, &graph);
    ExecutionExecutor executor(graph, &BackendRegistry::Default(), &context);
    for (const auto& name : taps) context.registerTap(name);
    context.clearDispatchTrace();

    const size_t context_len = std::max<size_t>(1, graph.modelConfig().context_length);
    size_t pos = 0;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (pos >= context_len) {
            if (note_out) {
                std::ostringstream oss;
                oss << "context length (" << context_len << ") exceeded during prefill at token "
                    << i << "; stopping prefill early";
                *note_out = oss.str();
            }
            break;
        }
        context.setToken(tokens[i]);
        context.setSequencePosition(pos);
        for (const auto& tname : {std::string("tokens"), std::string("token_ids")}) {
            if (graph.tensors().count(tname)) {
                context.setTensor(tname, {static_cast<float>(tokens[i])});
            }
        }
        auto res = executor.run();
        if (!res.success) {
            std::string last;
            for (const auto& t : res.trace) {
                if (!t.success) {
                    last = formatTraceEntry(t);
                    break;
                }
            }
            throw std::runtime_error("Execution failed at prefill step " + std::to_string(i) +
                                     (last.empty() ? "" : (": " + last)));
        }
        pos += 1;
    }

    if (dispatch_trace_out) {
        *dispatch_trace_out = context.dispatchTrace();
    }
    return context.tapData();
}

// Builds the "harness warning" lines describing nodes whose actual backend
// did not match what the caller pinned. Emits a leading summary line + up to
// kMaxListed offending node names. Returns the number of offenders.
size_t summarizeDispatchLeaks(const std::unordered_map<std::string, BackendKind>& trace,
                              BackendKind expected,
                              const std::string& warning_prefix,
                              std::vector<std::string>& notes_out) {
    std::vector<std::string> offenders;
    for (const auto& [name, kind] : trace) {
        if (kind != expected) offenders.push_back(name);
    }
    if (offenders.empty()) return 0;
    std::sort(offenders.begin(), offenders.end());
    constexpr size_t kMaxListed = 6;
    std::ostringstream head;
    head << warning_prefix << ": " << offenders.size() << " of "
         << trace.size() << " nodes";
    notes_out.push_back(head.str());
    size_t limit = std::min(offenders.size(), kMaxListed);
    for (size_t i = 0; i < limit; ++i) {
        notes_out.push_back("    - " + offenders[i]);
    }
    if (offenders.size() > kMaxListed) {
        std::ostringstream tail;
        tail << "    ... (" << (offenders.size() - kMaxListed) << " more)";
        notes_out.push_back(tail.str());
    }
    return offenders.size();
}

LayerComparison computeMetrics(const std::string& name,
                               const std::vector<float>* a,
                               const std::vector<float>* b) {
    LayerComparison c;
    c.name = name;
    c.present_a = (a != nullptr);
    c.present_b = (b != nullptr);
    if (!a || !b) return c;
    if (a->size() != b->size()) {
        c.element_count = std::min(a->size(), b->size());
        c.max_abs_diff = std::numeric_limits<float>::quiet_NaN();
        c.mean_abs_diff = std::numeric_limits<float>::quiet_NaN();
        c.rms = std::numeric_limits<float>::quiet_NaN();
        c.cosine_sim = std::numeric_limits<float>::quiet_NaN();
        return c;
    }

    c.element_count = a->size();
    if (c.element_count == 0) return c;

    double max_abs = 0.0, sum_abs = 0.0, sum_sq = 0.0;
    double dot = 0.0, na2 = 0.0, nb2 = 0.0;
    for (size_t i = 0; i < c.element_count; ++i) {
        double av = (*a)[i];
        double bv = (*b)[i];
        double d = av - bv;
        double ad = std::fabs(d);
        if (ad > max_abs) max_abs = ad;
        sum_abs += ad;
        sum_sq += d * d;
        dot += av * bv;
        na2 += av * av;
        nb2 += bv * bv;
    }
    c.max_abs_diff = static_cast<float>(max_abs);
    c.mean_abs_diff = static_cast<float>(sum_abs / static_cast<double>(c.element_count));
    c.rms = static_cast<float>(std::sqrt(sum_sq / static_cast<double>(c.element_count)));
    if (na2 > 0.0 && nb2 > 0.0) {
        c.cosine_sim = static_cast<float>(dot / (std::sqrt(na2) * std::sqrt(nb2)));
    } else {
        c.cosine_sim = std::numeric_limits<float>::quiet_NaN();
    }

    if (name.find("logits") != std::string::npos) {
        c.has_topk = true;
        auto topK = [&](const std::vector<float>& x, int k) {
            int kk = std::min<int>(k, static_cast<int>(x.size()));
            std::vector<size_t> idx(x.size());
            std::iota(idx.begin(), idx.end(), 0);
            std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                              [&x](size_t i, size_t j) { return x[i] > x[j]; });
            std::unordered_set<size_t> set(idx.begin(), idx.begin() + kk);
            return set;
        };
        auto overlap = [&](int k) {
            auto sa = topK(*a, k);
            auto sb = topK(*b, k);
            int n = 0;
            for (auto v : sa) if (sb.count(v)) ++n;
            return n;
        };
        c.top1_overlap = overlap(1);
        c.top5_overlap = overlap(5);
        c.top10_overlap = overlap(10);
    }
    return c;
}

std::vector<float> readReferenceTensor(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open reference tensor: " + path);
    }
    in.seekg(0, std::ios::end);
    std::streamsize bytes = in.tellg();
    in.seekg(0, std::ios::beg);
    if (bytes <= 0 || (bytes % static_cast<std::streamsize>(sizeof(float))) != 0) {
        throw std::runtime_error("Reference file size not a multiple of float32: " + path);
    }
    std::vector<float> values(static_cast<size_t>(bytes / sizeof(float)));
    in.read(reinterpret_cast<char*>(values.data()), bytes);
    if (!in) {
        throw std::runtime_error("Failed reading reference tensor: " + path);
    }
    return values;
}

} // namespace

std::string sanitizeForFilename(const std::string& name) {
    std::string out;
    out.reserve(name.size());
    for (char c : name) {
        out.push_back((c == '.' || c == '/') ? '_' : c);
    }
    return out;
}

CompareReport compareMetalVsCpu(const CompareOptions& opts) {
    CompareReport report;
    report.side_a_label = "cpu";
    report.side_b_label = "metal";

    if (opts.gguf_path.empty()) {
        throw std::runtime_error("compareMetalVsCpu: gguf_path is empty");
    }
    if (opts.prompt.empty()) {
        throw std::runtime_error("compareMetalVsCpu: prompt is empty");
    }

    Session session(opts.gguf_path);
    auto tokens = tokenize(session, opts.prompt, opts.wrap_chat_template);
    if (tokens.empty()) {
        throw std::runtime_error("Tokenizer produced no tokens for prompt");
    }
    {
        std::ostringstream oss;
        oss << "tokenized prompt -> " << tokens.size() << " tokens";
        report.notes.push_back(oss.str());
    }

    const auto& patterns = opts.tap_patterns.empty() ? defaultTapPatterns() : opts.tap_patterns;

    // Run A: CPU pinned.
    const bool prior_force_cpu = KernelDescriptorRegistry::forceCpu();
    KernelDescriptorRegistry::setForceCpu(true);
    std::vector<std::string> tap_names_cpu;
    std::string cpu_note;
    std::unordered_map<std::string, BackendKind> cpu_dispatch_trace;
    auto cpu_taps = runPrefillAndCaptureTaps(session, tokens, patterns, &tap_names_cpu, &cpu_note,
                                             &cpu_dispatch_trace);
    if (!cpu_note.empty()) report.notes.push_back("cpu: " + cpu_note);

    // Run B: Metal default.
    KernelDescriptorRegistry::setForceCpu(false);
    std::vector<std::string> tap_names_metal;
    std::string metal_note;
    std::unordered_map<std::string, BackendKind> metal_dispatch_trace;
    auto metal_taps = runPrefillAndCaptureTaps(session, tokens, patterns, &tap_names_metal, &metal_note,
                                               &metal_dispatch_trace);
    if (!metal_note.empty()) report.notes.push_back("metal: " + metal_note);

    KernelDescriptorRegistry::setForceCpu(prior_force_cpu);

    // Dispatch-trace audit. The side-A run is the load-bearing one: any node
    // that actually ran on Metal here means force-CPU leaked, which would
    // silently invalidate the CPU column. Treat as a real warning (and as
    // an error under MLC_HARNESS_STRICT).
    size_t cpu_leaks = summarizeDispatchLeaks(
        cpu_dispatch_trace,
        BackendKind::CPU,
        "HARNESS WARNING [dispatch leak]: nodes ran on Metal under force_cpu — comparison is invalid for these",
        report.notes);
    // Side-B is purely informational: nodes that ran on CPU under default
    // dispatch typically mean the Metal kernel is missing for that op shape,
    // not a dispatch bug. Phrased accordingly.
    summarizeDispatchLeaks(
        metal_dispatch_trace,
        BackendKind::Metal,
        "HARNESS NOTE [missing-kernel fallback]: nodes ran on CPU when Metal was expected (probably no Metal kernel for this op)",
        report.notes);

    if (cpu_leaks > 0 && std::getenv("MLC_HARNESS_STRICT") != nullptr) {
        report.strict_violation = true;
        report.success = false;
        report.notes.push_back("MLC_HARNESS_STRICT=1: failing because of dispatch leak above");
    }

    // The CPU run determines the canonical ordering — its tap_names_cpu is already
    // sorted by topo position. Use it; any tensors only present on the metal side
    // get appended at the end.
    std::vector<std::string> ordered = tap_names_cpu;
    std::unordered_set<std::string> seen(ordered.begin(), ordered.end());
    for (const auto& n : tap_names_metal) {
        if (!seen.count(n)) {
            ordered.push_back(n);
            seen.insert(n);
        }
    }

    for (const auto& name : ordered) {
        const std::vector<float>* a = nullptr;
        const std::vector<float>* b = nullptr;
        auto ita = cpu_taps.find(name);
        if (ita != cpu_taps.end()) a = &ita->second;
        auto itb = metal_taps.find(name);
        if (itb != metal_taps.end()) b = &itb->second;
        report.layers.push_back(computeMetrics(name, a, b));
    }

    return report;
}

CompareReport compareVsLlamaCpp(const CompareOptions& opts) {
    CompareReport report;
    report.side_a_label = "mlc-cpu";
    report.side_b_label = "llamacpp";

    if (opts.gguf_path.empty()) throw std::runtime_error("compareVsLlamaCpp: gguf_path is empty");
    if (opts.prompt.empty()) throw std::runtime_error("compareVsLlamaCpp: prompt is empty");
    if (opts.reference_dir.empty()) {
        throw std::runtime_error("compareVsLlamaCpp: reference_dir is required");
    }

    Session session(opts.gguf_path);
    auto tokens = tokenize(session, opts.prompt, opts.wrap_chat_template);
    if (tokens.empty()) {
        throw std::runtime_error("Tokenizer produced no tokens for prompt");
    }
    report.notes.push_back("tokenized prompt -> " + std::to_string(tokens.size()) + " tokens");
    report.notes.push_back("expected reference layout: <reference_dir>/<sanitized_tensor_name>.f32.bin "
                           "(raw little-endian float32, length matching the tensor)");

    const auto& patterns = opts.tap_patterns.empty() ? defaultTapPatterns() : opts.tap_patterns;

    const bool prior_force_cpu = KernelDescriptorRegistry::forceCpu();
    KernelDescriptorRegistry::setForceCpu(true);  // CPU = canonical for parity
    std::vector<std::string> tap_names;
    std::string note;
    auto mlc_taps = runPrefillAndCaptureTaps(session, tokens, patterns, &tap_names, &note);
    if (!note.empty()) report.notes.push_back("mlc-cpu: " + note);
    KernelDescriptorRegistry::setForceCpu(prior_force_cpu);

    // Diagnostic hook: write mlc-CPU's tap snapshots to disk if requested. Used by
    // ad-hoc analyses (per-head cosine, layout checks) that need raw tensor bytes.
    if (const char* dump_dir = std::getenv("MLC_PARITY_DUMP_SIDE_A")) {
        for (const auto& [name, data] : mlc_taps) {
            std::string path = std::string(dump_dir) + "/" + sanitizeForFilename(name) + ".f32.bin";
            std::ofstream out(path, std::ios::binary);
            if (out) {
                out.write(reinterpret_cast<const char*>(data.data()),
                          static_cast<std::streamsize>(data.size() * sizeof(float)));
            }
        }
        report.notes.push_back(std::string("dumped mlc-cpu taps to ") + dump_dir);
    }

    for (const auto& name : tap_names) {
        std::string ref_path = opts.reference_dir + "/" + sanitizeForFilename(name) + ".f32.bin";
        std::vector<float> ref;
        bool ref_ok = false;
        try {
            ref = readReferenceTensor(ref_path);
            ref_ok = true;
        } catch (const std::exception& e) {
            report.notes.push_back(std::string("missing/invalid reference for ") + name + ": " + e.what());
        }
        const std::vector<float>* a = nullptr;
        const std::vector<float>* b = nullptr;
        auto it = mlc_taps.find(name);
        if (it != mlc_taps.end()) a = &it->second;
        if (ref_ok) b = &ref;
        report.layers.push_back(computeMetrics(name, a, b));
    }

    return report;
}

namespace {
std::string fmtSci(float v) {
    if (std::isnan(v)) return "nan";
    char buf[24];
    std::snprintf(buf, sizeof(buf), "%.3e", v);
    return buf;
}
std::string fmtFixed(float v, int prec = 6) {
    if (std::isnan(v)) return "nan";
    char buf[24];
    std::snprintf(buf, sizeof(buf), "%.*f", prec, v);
    return buf;
}
} // namespace

void printTable(const CompareReport& report, std::ostream& out) {
    if (!report.notes.empty()) {
        for (const auto& n : report.notes) out << "# " << n << "\n";
    }
    out << "compare: side_a=" << report.side_a_label
        << " side_b=" << report.side_b_label << "\n";

    // Compute name column width.
    size_t name_w = 6;  // header "tensor"
    for (const auto& l : report.layers) name_w = std::max(name_w, l.name.size());
    name_w = std::max<size_t>(name_w, 24);

    auto pad = [](const std::string& s, size_t w) {
        if (s.size() >= w) return s;
        return s + std::string(w - s.size(), ' ');
    };
    auto rpad_num = [](const std::string& s, size_t w) {
        if (s.size() >= w) return s;
        return std::string(w - s.size(), ' ') + s;
    };

    std::ostringstream header;
    header << pad("tensor", name_w) << " | "
           << rpad_num("size", 8) << " | "
           << rpad_num("max_abs", 11) << " | "
           << rpad_num("mean_abs", 11) << " | "
           << rpad_num("rms", 11) << " | "
           << rpad_num("cosine", 10) << " | "
           << rpad_num("top1", 5) << " | "
           << rpad_num("top5", 5) << " | "
           << rpad_num("top10", 6);
    std::string head = header.str();
    out << head << "\n" << std::string(head.size(), '-') << "\n";

    for (const auto& l : report.layers) {
        std::string size_s, max_s, mean_s, rms_s, cos_s;
        std::string top1_s = "-", top5_s = "-", top10_s = "-";
        if (!l.present_a || !l.present_b) {
            size_s = "-";
            max_s = mean_s = rms_s = cos_s = (l.present_a ? "miss-b" : "miss-a");
        } else {
            size_s = std::to_string(l.element_count);
            max_s = fmtSci(l.max_abs_diff);
            mean_s = fmtSci(l.mean_abs_diff);
            rms_s = fmtSci(l.rms);
            cos_s = fmtFixed(l.cosine_sim, 6);
            if (l.has_topk) {
                top1_s = std::to_string(l.top1_overlap) + "/1";
                top5_s = std::to_string(l.top5_overlap) + "/5";
                top10_s = std::to_string(l.top10_overlap) + "/10";
            }
        }
        out << pad(l.name, name_w) << " | "
            << rpad_num(size_s, 8) << " | "
            << rpad_num(max_s, 11) << " | "
            << rpad_num(mean_s, 11) << " | "
            << rpad_num(rms_s, 11) << " | "
            << rpad_num(cos_s, 10) << " | "
            << rpad_num(top1_s, 5) << " | "
            << rpad_num(top5_s, 5) << " | "
            << rpad_num(top10_s, 6) << "\n";
    }
}

bool writeCsv(const CompareReport& report, const std::string& path) {
    if (path.empty()) return false;
    std::ofstream out(path, std::ios::trunc);
    if (!out) return false;
    out << "tensor,element_count,max_abs_diff,mean_abs_diff,rms,cosine_sim,"
           "top1_overlap_of_1,top5_overlap_of_5,top10_overlap_of_10,present_side_a,present_side_b\n";
    for (const auto& l : report.layers) {
        out << l.name << "," << l.element_count << ",";
        if (l.present_a && l.present_b) {
            out << fmtSci(l.max_abs_diff) << "," << fmtSci(l.mean_abs_diff) << ","
                << fmtSci(l.rms) << "," << fmtFixed(l.cosine_sim, 6) << ",";
            if (l.has_topk) {
                out << l.top1_overlap << "," << l.top5_overlap << "," << l.top10_overlap << ",";
            } else {
                out << ",,," ;
            }
        } else {
            out << ",,,,,,,";
        }
        out << (l.present_a ? "true" : "false") << ","
            << (l.present_b ? "true" : "false") << "\n";
    }
    return true;
}

} // namespace parity
} // namespace runtime
} // namespace mlc
