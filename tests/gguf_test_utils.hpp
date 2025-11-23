#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace mlc {
namespace test {
namespace gguf {

template<typename T>
inline void writeScalar(std::ofstream& file, T value) {
    file.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

inline void writeU8(std::ofstream& file, uint8_t value) { writeScalar(file, value); }
inline void writeU16(std::ofstream& file, uint16_t value) { writeScalar(file, value); }
inline void writeU32(std::ofstream& file, uint32_t value) { writeScalar(file, value); }
inline void writeU64(std::ofstream& file, uint64_t value) { writeScalar(file, value); }
inline void writeI8(std::ofstream& file, int8_t value) { writeScalar(file, value); }
inline void writeI16(std::ofstream& file, int16_t value) { writeScalar(file, value); }
inline void writeI32(std::ofstream& file, int32_t value) { writeScalar(file, value); }
inline void writeI64(std::ofstream& file, int64_t value) { writeScalar(file, value); }
inline void writeF32(std::ofstream& file, float value) { writeScalar(file, value); }
inline void writeF64(std::ofstream& file, double value) { writeScalar(file, value); }

inline void writeString(std::ofstream& file, const std::string& str) {
    writeU64(file, static_cast<uint64_t>(str.size()));
    if (!str.empty()) {
        file.write(str.c_str(), str.size());
    }
}

inline void writeBytes(std::ofstream& file, const std::vector<uint8_t>& data) {
    if (!data.empty()) {
        file.write(reinterpret_cast<const char*>(data.data()), data.size());
    }
}

inline void writeRawBlob(std::ofstream& file, const std::vector<uint8_t>& data) {
    writeU64(file, static_cast<uint64_t>(data.size()));
    writeBytes(file, data);
}

inline void writeRepeatedByte(std::ofstream& file, uint64_t length, uint8_t value) {
    std::vector<char> chunk(4096, static_cast<char>(value));
    while (length > 0) {
        size_t to_write = static_cast<size_t>(std::min<uint64_t>(length, chunk.size()));
        file.write(chunk.data(), to_write);
        length -= to_write;
    }
}

} // namespace gguf
} // namespace test
} // namespace mlc

