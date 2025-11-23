#pragma once

#include "runtime/execution_graph.hpp"
#include "frontends/gguf_loader.hpp"

namespace mlc {
namespace runtime {

class ExecutionPlanBuilder {
public:
    static ExecutionGraph BuildFromLoader(const frontend::GGUFLoader& loader);
    static ExecutionGraph BuildToy(size_t num_layers, size_t hidden_size);

private:
    static ExecutionGraph Build(const ModelConfig& config);
    static bool valueToSize(const frontend::GGUFValue& value, size_t& out);
    static std::string toLower(const std::string& s);
    static size_t readSize(const frontend::GGUFLoader& loader,
                           const std::vector<std::string>& keys,
                           size_t default_value);
    static size_t findSizeByKeywords(const frontend::GGUFLoader& loader,
                                     const std::vector<std::string>& keywords);
    static size_t inferVocabFromTokens(const frontend::GGUFLoader& loader);
};

} // namespace runtime
} // namespace mlc
