#pragma once

#include "frontends/gguf_loader.hpp"

#include <cstdint>
#include <string>
#include <limits>
#include <unordered_map>
#include <vector>
#include <set>

struct llama_model;
struct llama_vocab;

namespace mlc {
namespace runtime {

struct TokenizerConfig {
    bool add_bos = true;
    bool add_eos = false;
};

// Minimal GGUF tokenizer with Byte-Pair Encoding (BPE) support.
class Tokenizer {
public:
    Tokenizer() = default;
    explicit Tokenizer(const frontend::GGUFLoader& loader);
    ~Tokenizer();

    bool valid() const { return !tokens_.empty(); }

    std::vector<uint64_t> encode(const std::string& text,
                                 const TokenizerConfig& cfg = {}) const;

    std::string decode(const std::vector<uint64_t>& tokens) const;
    std::string tokenString(uint64_t id) const;
    bool isControlToken(uint64_t id) const;
    bool isEogToken(uint64_t id) const;
    size_t vocabSize() const { return tokens_.size(); }

    uint64_t bosId() const { return bos_id_; }
    uint64_t eosId() const { return eos_id_; }
    uint64_t unkId() const { return unk_id_; }

private:
    void buildIndices();
    std::vector<std::string> encodeBytes(const std::string& text) const;
    std::vector<uint64_t> bpeEncode(const std::string& text) const;
    std::vector<uint64_t> unigramEncode(const std::string& text) const;
    uint64_t unkOr(uint64_t id) const;
    static std::string decodeTokenString(const std::string& raw);

    void tryInitLlamaCpp(const frontend::GGUFLoader& loader);
    std::vector<uint64_t> encodeWithLlama(const std::string& text, bool add_special) const;
    std::string decodeWithLlama(const std::vector<uint64_t>& tokens) const;

    std::vector<std::string> tokens_;
    std::vector<std::string> token_bytes_;
    std::vector<float> scores_;
    std::string tokenizer_model_;
    std::unordered_map<std::string, uint64_t> token_to_id_;
    // Pair ranks for BPE merges: "tokA\tokB" -> rank
    std::unordered_map<std::string, size_t> merge_ranks_;

    // Fallback greedy index (when merges are missing).
    std::unordered_map<uint8_t, std::vector<uint64_t>> bucket_;
    uint64_t bos_id_ = std::numeric_limits<uint64_t>::max();
    uint64_t eos_id_ = std::numeric_limits<uint64_t>::max();
    uint64_t unk_id_ = std::numeric_limits<uint64_t>::max();

    // llama.cpp integration
    struct ::llama_model* llama_model_ = nullptr;
    const struct ::llama_vocab* llama_vocab_ = nullptr;
    bool use_llama_cpp_ = false;
};

} // namespace runtime
} // namespace mlc
