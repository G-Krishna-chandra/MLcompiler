#pragma once

#include <memory>

#include "frontends/gguf_loader.hpp"
#include "ir/ir.hpp"

namespace mlc {
namespace ir {

class IRBuilder {
public:
    static std::unique_ptr<Graph> BuildFromLoader(const frontend::GGUFLoader& loader);
};

} // namespace ir
} // namespace mlc
