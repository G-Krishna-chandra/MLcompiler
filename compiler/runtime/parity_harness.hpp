#pragma once

#include <iosfwd>
#include <string>
#include <vector>

namespace mlc {
namespace runtime {

namespace parity {

struct CompareOptions {
    std::string gguf_path;
    std::string prompt;
    // Substring patterns matched against execution-graph tensor names.
    // Empty -> defaults to embedding/per-block/final-norm/logits boundaries.
    std::vector<std::string> tap_patterns;
    // CSV output destination. Empty -> none.
    std::string csv_path;
    // For vs-llamacpp mode only: directory of reference tensor dumps.
    // Each tensor "X" is expected at "<dir>/<sanitized(X)>.f32.bin",
    // raw little-endian float32, length must match.
    std::string reference_dir;
    // Use chat-template wrap (TinyLlama [INST] ... [/INST]) around the prompt
    // when tokenizing. Default false (raw prompt) for parity tests.
    bool wrap_chat_template = false;
};

struct LayerComparison {
    std::string name;
    size_t element_count = 0;
    bool present_a = false;
    bool present_b = false;
    float max_abs_diff = 0.0f;
    float mean_abs_diff = 0.0f;
    float rms = 0.0f;
    float cosine_sim = 0.0f;
    bool has_topk = false;
    // For tensors whose name contains "logits".
    int top1_overlap = 0;
    int top5_overlap = 0;
    int top10_overlap = 0;
};

struct CompareReport {
    // Identifies the two sides for column headers.
    std::string side_a_label;
    std::string side_b_label;
    std::vector<LayerComparison> layers;
    // Diagnostic metadata captured during the run.
    std::vector<std::string> notes;
    bool success = true;
    // Set to true when MLC_HARNESS_STRICT is set and the side-A (force-CPU)
    // run had nodes that actually ran on Metal — i.e. a dispatch leak that
    // would silently invalidate the comparison.
    bool strict_violation = false;
};

// Run the harness in metal-vs-cpu mode (in-process; rebuilds the execution
// graph twice with the force-CPU flag flipped between runs).
CompareReport compareMetalVsCpu(const CompareOptions& opts);

// Run the harness in vs-llamacpp mode: executes mlc once (CPU pinned for
// reproducibility), reads reference tensor dumps from opts.reference_dir,
// and compares element-wise.
CompareReport compareVsLlamaCpp(const CompareOptions& opts);

// Render the report as an ASCII table to the given stream.
void printTable(const CompareReport& report, std::ostream& out);

// Append the report to a CSV file (creates with header if missing).
// Returns true on success.
bool writeCsv(const CompareReport& report, const std::string& path);

// Convert a tensor name to a filesystem-safe filename (used by reference dumps).
std::string sanitizeForFilename(const std::string& name);

} // namespace parity
} // namespace runtime
} // namespace mlc
