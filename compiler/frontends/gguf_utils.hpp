#pragma once

#include <cstdint>
#include <string>

namespace mlc {
namespace frontend {

// Convert GGUF dtype identifiers to human-readable strings
std::string ggufDtypeToString(uint32_t gguf_dtype);

// Attempt to parse a string like "Q4_0" into a GGML dtype constant.
// Returns true on success.
bool tryParseGGUFDtypeString(const std::string& value, uint32_t& out_dtype);

// Returns true if the GGML dtype encodes quantized weights/activations.
bool isQuantizedGGMLType(uint32_t gguf_dtype);

} // namespace frontend
} // namespace mlc
