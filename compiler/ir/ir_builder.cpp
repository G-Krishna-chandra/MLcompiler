#include "ir/ir_builder.hpp"

#include "frontends/gguf_to_ir.hpp"
#include "runtime/execution_graph.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <cctype>

namespace mlc {
namespace ir {

namespace {

size_t readSize(const frontend::GGUFLoader& loader,
                const std::vector<std::string>& keys,
                size_t fallback) {
    const auto& kv = loader.kvMetadata();
    for (const auto& key : keys) {
        auto it = kv.find(key);
        if (it == kv.end()) continue;
        const auto& value = it->second;
        switch (value.type) {
            case frontend::GGUFValueType::UINT32:
                return static_cast<size_t>(std::get<uint32_t>(value.data));
            case frontend::GGUFValueType::UINT64:
                return static_cast<size_t>(std::get<uint64_t>(value.data));
            case frontend::GGUFValueType::INT32: {
                int32_t v = std::get<int32_t>(value.data);
                if (v > 0) return static_cast<size_t>(v);
                break;
            }
            case frontend::GGUFValueType::INT64: {
                int64_t v = std::get<int64_t>(value.data);
                if (v > 0) return static_cast<size_t>(v);
                break;
            }
            default:
                break;
        }
    }
    return fallback;
}

std::string selectEmbeddingTensor(const frontend::GGUFLoader& loader,
                                  size_t hidden_size) {
    static const std::vector<std::string> candidates = {
        "tok_embeddings.weight",
        "token_embd.weight",
        "token_emb.weight",
        "token_embeddings.weight",
        "embed_tokens.weight",
        "model.embed_tokens.weight"
    };
    for (const auto& name : candidates) {
        auto it = loader.tensors().find(name);
        if (it == loader.tensors().end()) continue;
        if (it->second.shape.size() != 2) continue;
        if (it->second.shape[1] == hidden_size || it->second.shape[0] == hidden_size) {
            return name;
        }
    }
    for (const auto& [name, tensor] : loader.tensors()) {
        if (tensor.shape.size() != 2) continue;
        if (tensor.shape[0] == hidden_size || tensor.shape[1] == hidden_size) {
            return name;
        }
    }
    throw std::runtime_error("Unable to locate embedding tensor for IR builder");
}

ir::Tensor* lookupWeight(ir::Graph& graph, const std::string& name) {
    return graph.findTensor(name);
}

Tensor* addActivation(ir::Graph& graph,
                      const std::string& name,
                      const std::vector<int64_t>& shape,
                      ir::DataType dtype = ir::DataType::F32) {
    return graph.addTensor(name, shape, dtype);
}

runtime::ArchitectureFamily detectFamily(const std::string& arch) {
    if (arch.empty()) return runtime::ArchitectureFamily::Unknown;
    std::string lower = arch;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (lower.find("llama") != std::string::npos) return runtime::ArchitectureFamily::Llama;
    if (lower.find("gemma") != std::string::npos) return runtime::ArchitectureFamily::Gemma;
    if (lower.find("mistral") != std::string::npos) return runtime::ArchitectureFamily::Mistral;
    return runtime::ArchitectureFamily::Unknown;
}

} // namespace

std::unique_ptr<Graph> IRBuilder::BuildFromLoader(const frontend::GGUFLoader& loader) {
    auto graph = frontend::GGUFToIR(loader);
    if (!graph) {
        throw std::runtime_error("GGUFToIR returned null graph");
    }

    // Prefer family-specific keys, then fall back to common/llama.
    size_t num_layers = readSize(loader,
                                 {"llama.block_count", "llama.layers", "n_layer", "num_hidden_layers",
                                  "gemma.block_count", "mistral.block_count"},
                                 0);
    size_t hidden_size = readSize(loader,
                                  {"llama.embedding_length", "llama.n_embd", "n_embd",
                                   "gemma.embedding_length", "mistral.embedding_length"},
                                  0);
    size_t head_count = readSize(loader,
                                 {"llama.attention.head_count", "attention.head_count", "num_attention_heads",
                                  "gemma.attention.head_count", "mistral.attention.head_count"},
                                 0);
    if (head_count == 0) head_count = 8;
    if (num_layers == 0 || hidden_size == 0) {
        throw std::runtime_error("IRBuilder requires block_count and embedding_length metadata");
    }

    Tensor* tokens = graph->addTensor("token_ids", {1}, ir::DataType::I4);
    tokens->metadata["role"] = "input_tokens";
    Node* token_source = graph->addNode(ir::OpKind::Reshape, "token_source");
    token_source->outputs.push_back(tokens);

    std::string embedding_name = selectEmbeddingTensor(loader, hidden_size);
    Tensor* embedding_weight = lookupWeight(*graph, embedding_name);
    if (!embedding_weight) {
        throw std::runtime_error("Embedding tensor '" + embedding_name + "' not imported into IR graph");
    }

    Tensor* hidden_tensor = addActivation(*graph,
                                          "hidden_state_0",
                                          {static_cast<int64_t>(hidden_size)});
    Node* embedding_node = graph->addNode(ir::OpKind::Embedding, "embedding_lookup");
    embedding_node->inputs.push_back(token_source);
    embedding_node->activation_inputs.push_back(tokens);
    embedding_node->tensor_inputs.push_back(embedding_weight);
    embedding_node->outputs.push_back(hidden_tensor);
    embedding_node->metadata["weight"] = embedding_name;
    embedding_node->attributes["hidden_size"] = static_cast<float>(hidden_size);

    Node* current_node = embedding_node;
    Tensor* current_tensor = hidden_tensor;

    std::string arch = loader.kvMetadata().count("general.architecture")
                           ? std::get<std::string>(loader.kvMetadata().at("general.architecture").data)
                           : "";
    runtime::ArchitectureFamily family = detectFamily(arch);
    std::string ffn_activation = (family == runtime::ArchitectureFamily::Gemma) ? "geglu" : "silu";

    for (size_t layer = 0; layer < num_layers; ++layer) {
        std::string prefix = "blk." + std::to_string(layer) + ".";
        std::string layer_tag = std::to_string(layer);

        Tensor* residual_input = current_tensor;
        Node* residual_node = current_node;

        auto buildLinear = [&](Node* input_node,
                               Tensor* activation,
                               const std::string& suffix,
                               const std::string& role) -> std::pair<Node*, Tensor*> {
            std::string weight_name = prefix + suffix + ".weight";
            Tensor* weight = lookupWeight(*graph, weight_name);
            if (!weight) return {nullptr, nullptr};
            Tensor* out_tensor = addActivation(*graph,
                                               prefix + suffix + ".out",
                                               {static_cast<int64_t>(hidden_size)});
            Node* node = graph->addNode(ir::OpKind::MatMul, prefix + suffix + ".matmul");
            if (input_node) node->inputs.push_back(input_node);
            if (activation) node->activation_inputs.push_back(activation);
            node->tensor_inputs.push_back(weight);
            node->outputs.push_back(out_tensor);
            node->metadata["layer"] = layer_tag;
            node->metadata["role"] = role;
            node->metadata["weight"] = weight_name;
            std::string bias_name = prefix + suffix + ".bias";
            Tensor* bias = lookupWeight(*graph, bias_name);
            if (bias) {
                node->tensor_inputs.push_back(bias);
                node->metadata["bias"] = bias_name;
            }
            return {node, out_tensor};
        };

        // Layer norm before attention if available.
        Tensor* norm_output = current_tensor;
        Node* norm_node = nullptr;
        if (Tensor* norm_weight = lookupWeight(*graph, prefix + "attn_norm.weight")) {
            norm_output = addActivation(*graph,
                                        prefix + "attn_norm.out",
                                        {static_cast<int64_t>(hidden_size)});
            norm_node = graph->addNode(ir::OpKind::LayerNorm, prefix + "attn_norm");
            norm_node->inputs.push_back(current_node);
            norm_node->activation_inputs.push_back(current_tensor);
            norm_node->tensor_inputs.push_back(norm_weight);
            norm_node->outputs.push_back(norm_output);
            norm_node->metadata["layer"] = layer_tag;
            norm_node->metadata["weight"] = prefix + "attn_norm.weight";
            current_node = norm_node;
            current_tensor = norm_output;
        }

        Node* attn_input_node = norm_node ? norm_node : current_node;
        Tensor* attn_input_tensor = norm_output;
        auto q = buildLinear(attn_input_node, attn_input_tensor, "attn_q", "attn_q");
        auto k = buildLinear(attn_input_node, attn_input_tensor, "attn_k", "attn_k");
        auto v = buildLinear(attn_input_node, attn_input_tensor, "attn_v", "attn_v");

        Tensor* attn_mix = addActivation(*graph,
                                         prefix + "attention_mix",
                                         {static_cast<int64_t>(hidden_size)});
        Node* attention = graph->addNode(ir::OpKind::Attention, prefix + "attention");
        if (q.first) attention->inputs.push_back(q.first);
        if (k.first) attention->inputs.push_back(k.first);
        if (v.first) attention->inputs.push_back(v.first);
        if (q.second) attention->activation_inputs.push_back(q.second);
        if (k.second) attention->activation_inputs.push_back(k.second);
        if (v.second) attention->activation_inputs.push_back(v.second);
        attention->outputs.push_back(attn_mix);
        attention->metadata["layer"] = layer_tag;
        attention->metadata["heads"] = std::to_string(head_count);
        size_t sliding_window = readSize(loader,
                                         {"attention.sliding_window", "mistral.sliding_window"},
                                         0);
        if (sliding_window > 0) {
            attention->metadata["sliding_window"] = std::to_string(sliding_window);
        }

        auto out = buildLinear(attention, attn_mix, "attn_output", "attn_out");
        Node* attn_out_node = out.first ? out.first : attention;
        Tensor* attn_out_tensor = out.second ? out.second : attn_mix;

        Tensor* resid1_tensor = addActivation(*graph,
                                              prefix + "residual_1",
                                              {static_cast<int64_t>(hidden_size)});
        Node* resid1 = graph->addNode(ir::OpKind::Add, prefix + "residual_add1");
        if (residual_node) resid1->inputs.push_back(residual_node);
        if (attn_out_node) resid1->inputs.push_back(attn_out_node);
        resid1->activation_inputs.push_back(residual_input);
        resid1->activation_inputs.push_back(attn_out_tensor);
        resid1->outputs.push_back(resid1_tensor);
        resid1->metadata["layer"] = layer_tag;
        resid1->metadata["role"] = "residual_1";

        Tensor* ffn_norm_out = resid1_tensor;
        Node* ffn_norm_node = resid1;
        if (Tensor* norm_weight = lookupWeight(*graph, prefix + "ffn_norm.weight")) {
            ffn_norm_out = addActivation(*graph,
                                         prefix + "ffn_norm.out",
                                         {static_cast<int64_t>(hidden_size)});
            ffn_norm_node = graph->addNode(ir::OpKind::LayerNorm, prefix + "ffn_norm");
            ffn_norm_node->inputs.push_back(resid1);
            ffn_norm_node->activation_inputs.push_back(resid1_tensor);
            ffn_norm_node->tensor_inputs.push_back(norm_weight);
            ffn_norm_node->outputs.push_back(ffn_norm_out);
            ffn_norm_node->metadata["layer"] = layer_tag;
            ffn_norm_node->metadata["weight"] = prefix + "ffn_norm.weight";
        }

        auto gate = buildLinear(ffn_norm_node, ffn_norm_out, "ffn_gate", "ffn_gate");
        auto up = buildLinear(ffn_norm_node, ffn_norm_out, "ffn_up", "ffn_up");
        Tensor* ffn_mix = addActivation(*graph,
                                        prefix + "ffn_mix",
                                        {static_cast<int64_t>(hidden_size)});
        Node* ffn = graph->addNode(ir::OpKind::FeedForward, prefix + "ffn");
        if (gate.first) ffn->inputs.push_back(gate.first);
        if (up.first) ffn->inputs.push_back(up.first);
        if (gate.second) ffn->activation_inputs.push_back(gate.second);
        if (up.second) ffn->activation_inputs.push_back(up.second);
        ffn->outputs.push_back(ffn_mix);
        ffn->metadata["layer"] = layer_tag;
        ffn->metadata["activation"] = ffn_activation;

        auto down = buildLinear(ffn, ffn_mix, "ffn_down", "ffn_down");

        Node* ffn_out_node = down.first ? down.first : ffn;
        Tensor* ffn_out_tensor = down.second ? down.second : ffn_mix;

        Tensor* resid2_tensor = addActivation(*graph,
                                              prefix + "residual_2",
                                              {static_cast<int64_t>(hidden_size)});
        Node* resid2 = graph->addNode(ir::OpKind::Add, prefix + "residual_add2");
        resid2->inputs.push_back(resid1);
        if (ffn_out_node) resid2->inputs.push_back(ffn_out_node);
        resid2->activation_inputs.push_back(resid1_tensor);
        resid2->activation_inputs.push_back(ffn_out_tensor);
        resid2->outputs.push_back(resid2_tensor);
        resid2->metadata["layer"] = layer_tag;
        resid2->metadata["role"] = "residual_2";

        current_node = resid2;
        current_tensor = resid2_tensor;
    }

    // Final normalization if available
    static const std::vector<std::string> final_norm_names = {
        "norm.weight",
        "output_norm.weight",
        "model.norm.weight"
    };
    for (const auto& norm_name : final_norm_names) {
        if (Tensor* norm_weight = lookupWeight(*graph, norm_name)) {
            Tensor* norm_output = addActivation(*graph,
                                                "final_norm_out",
                                                {static_cast<int64_t>(hidden_size)});
            Node* norm = graph->addNode(ir::OpKind::LayerNorm, "final_norm");
            norm->inputs.push_back(current_node);
            norm->activation_inputs.push_back(current_tensor);
            norm->tensor_inputs.push_back(norm_weight);
            norm->outputs.push_back(norm_output);
            norm->metadata["weight"] = norm_name;
            current_node = norm;
            current_tensor = norm_output;
            break;
        }
    }

    // LM head / logits
    static const std::vector<std::string> head_candidates = {
        "output.weight",
        "lm_head.weight",
        "model.output.weight",
        "model.lm_head.weight"
    };
    std::string head_name;
    for (const auto& candidate : head_candidates) {
        if (loader.tensors().count(candidate)) {
            head_name = candidate;
            break;
        }
    }
    if (head_name.empty()) head_name = head_candidates.front();
    Tensor* head_weight = lookupWeight(*graph, head_name);
    if (head_weight) {
        int64_t vocab = head_weight->shape.empty()
                            ? static_cast<int64_t>(hidden_size)
                            : static_cast<int64_t>(head_weight->shape[0]);
        Tensor* logits = addActivation(*graph,
                                       "logits",
                                       {vocab},
                                       ir::DataType::F32);
        Node* head = graph->addNode(ir::OpKind::Linear, "lm_head");
        head->inputs.push_back(current_node);
        head->activation_inputs.push_back(current_tensor);
        head->tensor_inputs.push_back(head_weight);
        head->outputs.push_back(logits);
        head->metadata["weight"] = head_name;
    }

    return graph;
}

} // namespace ir
} // namespace mlc
