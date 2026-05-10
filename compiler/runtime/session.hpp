#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include "frontends/gguf_loader.hpp"

namespace mlc {
namespace runtime {

class Session {
public:
    explicit Session(const std::string& gguf_path);

    const mlc::frontend::GGUFLoader& loader() const { return loader_; }

    // Executes a single linear layer (weight matrix times input vector).
    std::vector<float> runLinear(const std::string& tensor_name,
                                 const std::vector<float>& input) const;

    // Returns embedding vector for a specific token id from a 2D F32 tensor.
    std::vector<float> getEmbedding(const std::string& tensor_name,
                                    uint64_t token_id) const;

    // Returns cached raw tensor bytes.
    const std::vector<uint8_t>& tensorData(const mlc::frontend::GGUFTensorInfo& tensor) const;

private:
    mlc::frontend::GGUFLoader loader_;
    mutable std::unordered_map<std::string, std::vector<uint8_t>> raw_tensor_cache_;
};

} // namespace runtime
} // namespace mlc
