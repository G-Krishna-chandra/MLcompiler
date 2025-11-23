#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <variant>

namespace mlc {
namespace frontend {

enum class GGUFValueType : uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    INT64 = 6,
    UINT64 = 7,
    FLOAT32 = 8,
    FLOAT64 = 9,
    BOOL = 10,
    STRING = 11,
    ARRAY = 12,
    RAW = 13,      // legacy RAW blobs stored in KV metadata
    TENSOR = 14    // tensor directory entries (not used in KV pairs)
};

struct GGUFValue {
    GGUFValueType type;
    std::variant<
        uint8_t, int8_t,
        uint16_t, int16_t,
        uint32_t, int32_t,
        uint64_t, int64_t,
        float, double,
        bool,
        std::string,
        std::vector<GGUFValue>,     // for ARRAY type
        std::vector<uint8_t>        // for RAW bytes
    > data;
};

struct GGUFHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
};

struct GGUFTensorInfo {
    std::string name;
    std::vector<uint64_t> shape;  // GGUF v3 uses uint64_t for dimensions
    uint32_t dtype;
    uint64_t offset;
    uint64_t n_bytes;  // Size of tensor data in bytes
};

class GGUFLoader {
public:
    explicit GGUFLoader(const std::string& path);
    
    bool load();
    
    void setVerbose(bool verbose) { verbose_ = verbose; }
    bool verbose() const { return verbose_; }
    
    const std::unordered_map<std::string, GGUFTensorInfo>& tensors() const {
        return tensor_map_;
    }
    
    const std::unordered_map<std::string, GGUFValue>& kvMetadata() const {
        return kv_metadata_;
    }
    
    const GGUFHeader& header() const {
        return header_;
    }

    uint32_t quantizationVersion() const { return quant_version_; }
    
    std::vector<uint8_t> loadTensorData(const GGUFTensorInfo& t) const;

private:
    std::string path_;
    GGUFHeader header_;
    std::unordered_map<std::string, GGUFTensorInfo> tensor_map_;
    std::unordered_map<std::string, GGUFValue> kv_metadata_;  // Store KV metadata for introspection
    bool verbose_ = false;
    uint32_t quant_version_ = 1;
    size_t alignment_ = 32;
    
    // Helper functions for binary reading (using FILE* for alignment)
    uint8_t readU8(FILE* file);
    int8_t readI8(FILE* file);
    uint16_t readU16(FILE* file);
    int16_t readI16(FILE* file);
    uint32_t readU32(FILE* file);
    int32_t readI32(FILE* file);
    uint64_t readU64(FILE* file);
    int64_t readI64(FILE* file);
    float readF32(FILE* file);
    double readF64(FILE* file);
    bool readBool(FILE* file);
    std::string readString(FILE* file, bool enforce_soft_limit = false);
    std::vector<uint8_t> readRaw(FILE* file);
    std::vector<uint64_t> readU64Array(FILE* file);
    
    // Alignment helper
    void alignOffset(FILE* file, size_t alignment);
    
    // Parse a single GGUF value
    GGUFValue parseValue(FILE* file, GGUFValueType type);
    
    // Parse the key-value metadata section
    void parseKVSection(FILE* file);
    
    // Parse tensor directory
    void parseTensorDirectory(FILE* file, uint64_t tensor_data_start);
};

} // namespace frontend
} // namespace mlc
