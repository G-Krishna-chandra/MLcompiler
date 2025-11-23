#include "gguf_utils.hpp"
#include "frontends/ggml_types.hpp"

#include <cctype>

namespace mlc {
namespace frontend {

namespace {

bool equalsIgnoreCase(const std::string& lhs, const std::string& rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
        unsigned char a = static_cast<unsigned char>(lhs[i]);
        unsigned char b = static_cast<unsigned char>(rhs[i]);
        if (std::tolower(a) != std::tolower(b)) {
            return false;
        }
    }
    return true;
}

} // namespace

std::string ggufDtypeToString(uint32_t gguf_dtype) {
    switch (gguf_dtype) {
        case GGML_TYPE_F32: return "F32";
        case GGML_TYPE_F16: return "F16";
        case GGML_TYPE_BF16: return "BF16";
        case GGML_TYPE_Q4_0: return "Q4_0";
        case GGML_TYPE_Q4_1: return "Q4_1";
        case GGML_TYPE_Q5_0: return "Q5_0";
        case GGML_TYPE_Q5_1: return "Q5_1";
        case GGML_TYPE_Q8_0: return "Q8_0";
        case GGML_TYPE_Q8_1: return "Q8_1";
        case GGML_TYPE_Q2_K: return "Q2_K";
        case GGML_TYPE_Q3_K: return "Q3_K";
        case GGML_TYPE_Q4_K: return "Q4_K";
        case GGML_TYPE_Q5_K: return "Q5_K";
        case GGML_TYPE_Q6_K: return "Q6_K";
        case GGML_TYPE_Q8_K: return "Q8_K";
        case GGML_TYPE_I8: return "I8";
        case GGML_TYPE_I16: return "I16";
        case GGML_TYPE_I32: return "I32";
        default: return "UNKNOWN_" + std::to_string(gguf_dtype);
    }
}

bool tryParseGGUFDtypeString(const std::string& value, uint32_t& out_dtype) {
    if (value.empty()) return false;
    struct Mapping {
        const char* name;
        uint32_t type;
    };
    static constexpr Mapping kMappings[] = {
        {"F32", GGML_TYPE_F32},
        {"F16", GGML_TYPE_F16},
        {"BF16", GGML_TYPE_BF16},
        {"Q4_0", GGML_TYPE_Q4_0},
        {"Q4_1", GGML_TYPE_Q4_1},
        {"Q5_0", GGML_TYPE_Q5_0},
        {"Q5_1", GGML_TYPE_Q5_1},
        {"Q8_0", GGML_TYPE_Q8_0},
        {"Q8_1", GGML_TYPE_Q8_1},
        {"Q2_K", GGML_TYPE_Q2_K},
        {"Q3_K", GGML_TYPE_Q3_K},
        {"Q4_K", GGML_TYPE_Q4_K},
        {"Q5_K", GGML_TYPE_Q5_K},
        {"Q6_K", GGML_TYPE_Q6_K},
        {"Q8_K", GGML_TYPE_Q8_K},
        {"I8", GGML_TYPE_I8},
        {"I16", GGML_TYPE_I16},
        {"I32", GGML_TYPE_I32},
        {"I64", GGML_TYPE_I64},
    };
    for (const auto& mapping : kMappings) {
        if (equalsIgnoreCase(value, mapping.name)) {
            out_dtype = mapping.type;
            return true;
        }
    }
    return false;
}

bool isQuantizedGGMLType(uint32_t gguf_dtype) {
    switch (gguf_dtype) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_K:
            return true;
        default:
            return false;
    }
}

} // namespace frontend
} // namespace mlc
