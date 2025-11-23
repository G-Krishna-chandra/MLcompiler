#pragma once

#include <string>
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

private:
    mlc::frontend::GGUFLoader loader_;
};

} // namespace runtime
} // namespace mlc
