#include "runtime/tokenizer.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>

#ifdef MLC_ENABLE_LLAMA_TOKENIZER
#include "llama.h"
#endif
#include <limits>

namespace mlc {
namespace runtime {

namespace {
template <typename T>
uint64_t readId(const std::unordered_map<std::string, frontend::GGUFValue>& kv,
                const std::string& key,
                uint64_t fallback) {
    auto it = kv.find(key);
    if (it == kv.end()) return fallback;
    if constexpr (std::is_same<T, uint32_t>::value) {
        if (it->second.type == frontend::GGUFValueType::UINT32) {
            return static_cast<uint64_t>(std::get<uint32_t>(it->second.data));
        }
    } else if constexpr (std::is_same<T, uint64_t>::value) {
        if (it->second.type == frontend::GGUFValueType::UINT64) {
            return std::get<uint64_t>(it->second.data);
        }
    }
    return fallback;
}
} // namespace

Tokenizer::Tokenizer(const frontend::GGUFLoader& loader) {
    const auto& kv = loader.kvMetadata();
    tryInitLlamaCpp(loader);

    if (use_llama_cpp_) {
        // We still populate tokens_ metadata for compatibility with tests.
        const int n_vocab = llama_vocab_n_tokens(llama_vocab_);
        tokens_.resize(static_cast<size_t>(n_vocab));
        token_bytes_.resize(static_cast<size_t>(n_vocab));
        for (int i = 0; i < n_vocab; ++i) {
            const char* text = llama_vocab_get_text(llama_vocab_, i);
            if (text) {
                tokens_[i] = text;
                token_bytes_[i] = text;
            }
        }
        bos_id_ = llama_vocab_bos(llama_vocab_);
        eos_id_ = llama_vocab_eos(llama_vocab_);
        if (unk_id_ == std::numeric_limits<uint64_t>::max()) {
            unk_id_ = readId<uint32_t>(kv, "tokenizer.ggml.unk_token_id",
                                       readId<uint64_t>(kv, "tokenizer.ggml.unk_token_id",
                                                        readId<uint32_t>(kv, "tokenizer.ggml.unknown_token_id",
                                                                         readId<uint64_t>(kv, "tokenizer.ggml.unknown_token_id", unk_id_))));
        }
        buildIndices();
        return;
    }

    auto model_it = kv.find("tokenizer.ggml.model");
    if (model_it != kv.end() && model_it->second.type == frontend::GGUFValueType::STRING) {
        tokenizer_model_ = std::get<std::string>(model_it->second.data);
    }

    auto it = kv.find("tokenizer.ggml.tokens");
    if (it != kv.end() && it->second.type == frontend::GGUFValueType::ARRAY) {
        const auto& arr = std::get<std::vector<frontend::GGUFValue>>(it->second.data);
        tokens_.reserve(arr.size());
        token_bytes_.reserve(arr.size());
        for (const auto& v : arr) {
            if (v.type == frontend::GGUFValueType::STRING) {
                const auto& raw = std::get<std::string>(v.data);
                tokens_.push_back(raw);
                token_bytes_.push_back(decodeTokenString(raw));
            }
        }
    }

    auto score_it = kv.find("tokenizer.ggml.scores");
    if (score_it != kv.end() && score_it->second.type == frontend::GGUFValueType::ARRAY) {
        const auto& arr = std::get<std::vector<frontend::GGUFValue>>(score_it->second.data);
        scores_.reserve(arr.size());
        for (const auto& v : arr) {
            if (v.type == frontend::GGUFValueType::FLOAT32) {
                scores_.push_back(std::get<float>(v.data));
            } else if (v.type == frontend::GGUFValueType::FLOAT64) {
                scores_.push_back(static_cast<float>(std::get<double>(v.data)));
            }
        }
    }

    bos_id_ = readId<uint32_t>(kv, "tokenizer.ggml.bos_token_id",
                               readId<uint64_t>(kv, "tokenizer.ggml.bos_token_id", bos_id_));
    eos_id_ = readId<uint32_t>(kv, "tokenizer.ggml.eos_token_id",
                               readId<uint64_t>(kv, "tokenizer.ggml.eos_token_id", eos_id_));
    unk_id_ = readId<uint32_t>(kv, "tokenizer.ggml.unk_token_id",
                               readId<uint64_t>(kv, "tokenizer.ggml.unk_token_id", unk_id_));

    for (size_t i = 0; i < tokens_.size(); ++i) {
        token_to_id_[token_bytes_.empty() ? tokens_[i] : token_bytes_[i]] = static_cast<uint64_t>(i);
    }
    // Load BPE merges (if present).
    auto merge_it = kv.find("tokenizer.ggml.merges");
    if (merge_it != kv.end() && merge_it->second.type == frontend::GGUFValueType::ARRAY) {
        const auto& merges = std::get<std::vector<frontend::GGUFValue>>(merge_it->second.data);
        for (size_t rank = 0; rank < merges.size(); ++rank) {
            if (merges[rank].type != frontend::GGUFValueType::STRING) continue;
            const auto& merge = std::get<std::string>(merges[rank].data);
            // Merge strings are "a b"
            auto pos = merge.find(' ');
            if (pos == std::string::npos) continue;
            std::string key = merge;
            merge_ranks_[key] = rank;
        }
    }

    buildIndices();
}

Tokenizer::~Tokenizer() {
#ifdef MLC_ENABLE_LLAMA_TOKENIZER
    if (llama_model_) {
        llama_model_free(llama_model_);
        llama_model_ = nullptr;
    }
#endif
}

void Tokenizer::buildIndices() {
    bucket_.clear();
    for (size_t i = 0; i < token_bytes_.size(); ++i) {
        const std::string& tok = token_bytes_.empty() ? tokens_[i] : token_bytes_[i];
        if (tok.empty()) continue;
        uint8_t lead = static_cast<uint8_t>(tok[0]);
        bucket_[lead].push_back(static_cast<uint64_t>(i));
    }
    for (auto& kv : bucket_) {
        auto& vec = kv.second;
        std::sort(vec.begin(), vec.end(), [&](uint64_t a, uint64_t b) {
            const auto& ta = tokens_[a];
            const auto& tb = tokens_[b];
            if (ta.size() == tb.size()) return a < b;
            return ta.size() > tb.size(); // longest first
        });
    }
}

std::vector<uint64_t> Tokenizer::encode(const std::string& text,
                                        const TokenizerConfig& cfg) const {
    std::vector<uint64_t> ids;
    if (!valid()) return ids;

    if (use_llama_cpp_) {
        ids = encodeWithLlama(text, /*add_special=*/false);
        if (cfg.add_bos && bos_id_ != std::numeric_limits<uint64_t>::max() && (ids.empty() || ids.front() != bos_id_)) {
            ids.insert(ids.begin(), bos_id_);
        }
        if (cfg.add_eos && eos_id_ != std::numeric_limits<uint64_t>::max()) {
            ids.push_back(eos_id_);
        }
        return ids;
    }

    if (cfg.add_bos && bos_id_ != std::numeric_limits<uint64_t>::max()) {
        ids.push_back(bos_id_);
    }

    bool use_unigram = false;
    bool use_bpe = false;
    if (!tokenizer_model_.empty()) {
        if (tokenizer_model_ == "llama" || tokenizer_model_ == "spm" || tokenizer_model_ == "unigram") {
            use_unigram = true;
        } else if (tokenizer_model_ == "gpt2" || tokenizer_model_ == "bpe") {
            use_bpe = true;
        }
    }

    if (use_unigram) {
        if (scores_.size() == tokens_.size()) {
            auto uni_ids = unigramEncode(text);
            ids.insert(ids.end(), uni_ids.begin(), uni_ids.end());
        }
    } else if (use_bpe) {
        if (!merge_ranks_.empty()) {
            auto bpe_ids = bpeEncode(text);
            ids.insert(ids.end(), bpe_ids.begin(), bpe_ids.end());
        }
    } else if (!merge_ranks_.empty()) {
        auto bpe_ids = bpeEncode(text);
        ids.insert(ids.end(), bpe_ids.begin(), bpe_ids.end());
    } else if (scores_.size() == tokens_.size()) {
        auto uni_ids = unigramEncode(text);
        ids.insert(ids.end(), uni_ids.begin(), uni_ids.end());
    }

    if (ids.empty()) {
        // Fallback: greedy byte-prefix.
        size_t i = 0;
        while (i < text.size()) {
            uint8_t lead = static_cast<uint8_t>(text[i]);
            uint64_t matched_id = std::numeric_limits<uint64_t>::max();
            size_t matched_len = 0;
            auto it = bucket_.find(lead);
            if (it != bucket_.end()) {
                const auto& candidates = it->second;
                for (uint64_t cid : candidates) {
                    const std::string& tok = token_bytes_.empty() ? tokens_[cid] : token_bytes_[cid];
                    size_t len = tok.size();
                    if (len > text.size() - i) continue;
                    if (std::memcmp(text.data() + i, tok.data(), len) == 0) {
                        matched_id = cid;
                        matched_len = len;
                        break;
                    }
                }
            }
            if (matched_len == 0) {
                ids.push_back(unkOr(unk_id_));
                ++i;
            } else {
                ids.push_back(matched_id);
                i += matched_len;
            }
        }
    }

    if (cfg.add_eos && eos_id_ != std::numeric_limits<uint64_t>::max()) {
        ids.push_back(eos_id_);
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<uint64_t>& tokens) const {
    if (!valid()) return {};
    if (use_llama_cpp_) {
        return decodeWithLlama(tokens);
    }

    std::string out;
    const auto& vocab = token_bytes_.empty() ? tokens_ : token_bytes_;

    auto isSpecial = [](const std::string& tok) -> bool {
        return tok == "<s>" || tok == "</s>" || tok == "<unk>";
    };
    auto isSpUnderscore = [](const std::string& tok) -> bool {
        return tok.size() >= 3 &&
               static_cast<unsigned char>(tok[0]) == 0xE2 &&
               static_cast<unsigned char>(tok[1]) == 0x96 &&
               static_cast<unsigned char>(tok[2]) == 0x81;
    };

    for (uint64_t id : tokens) {
        if (id >= vocab.size()) continue;
        const std::string& tok = vocab[id];
        if (isSpecial(tok)) continue;
        if (isSpUnderscore(tok)) {
            // SentencePiece convention: leading "▁" marks a word boundary.
            std::string tail = tok.substr(3);
            if (!tail.empty()) {
                if (!out.empty()) out.push_back(' ');
                out.append(tail);
            }
        } else {
            out.append(tok);
        }
    }
    return out;
}

std::string Tokenizer::tokenString(uint64_t id) const {
    if (id >= tokens_.size()) return {};
    return tokens_[id];
}

bool Tokenizer::isControlToken(uint64_t id) const {
#ifdef MLC_ENABLE_LLAMA_TOKENIZER
    if (use_llama_cpp_ && llama_vocab_) {
        return llama_vocab_is_control(llama_vocab_, static_cast<llama_token>(id));
    }
#endif
    if (id >= tokens_.size()) return false;
    const std::string& tok = tokens_[id];
    if (tok == "<s>" || tok == "</s>" || tok == "<unk>") return true;
    if (tok.size() >= 3 && tok[0] == '<' && tok[1] == '|' && tok.back() == '>') {
        return true;
    }
    return false;
}

bool Tokenizer::isEogToken(uint64_t id) const {
#ifdef MLC_ENABLE_LLAMA_TOKENIZER
    if (use_llama_cpp_ && llama_vocab_) {
        return llama_vocab_is_eog(llama_vocab_, static_cast<llama_token>(id));
    }
#endif
    return id == eos_id_;
}

std::vector<std::string> Tokenizer::encodeBytes(const std::string& text) const {
    std::vector<std::string> bytes;
    bytes.reserve(text.size());
    for (unsigned char c : text) {
        bytes.emplace_back(1, static_cast<char>(c));
    }
    return bytes;
}

uint64_t Tokenizer::unkOr(uint64_t id) const {
    if (id != std::numeric_limits<uint64_t>::max()) return id;
    return 0; // best-effort fallback
}

std::vector<uint64_t> Tokenizer::bpeEncode(const std::string& text) const {
    if (text.empty()) return {};
    auto symbols = encodeBytes(text);
    if (symbols.empty()) return {};

    auto get_pair_rank = [&](const std::string& a, const std::string& b) -> size_t {
        std::string key;
        key.reserve(a.size() + b.size() + 1);
        key.append(a);
        key.push_back(' ');
        key.append(b);
        auto it = merge_ranks_.find(key);
        if (it == merge_ranks_.end()) return std::numeric_limits<size_t>::max();
        return it->second;
    };

    while (symbols.size() > 1) {
        size_t best_rank = std::numeric_limits<size_t>::max();
        size_t best_index = symbols.size();
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            size_t rank = get_pair_rank(symbols[i], symbols[i + 1]);
            if (rank < best_rank) {
                best_rank = rank;
                best_index = i;
            }
        }
        if (best_rank == std::numeric_limits<size_t>::max() || best_index >= symbols.size() - 1) {
            break;
        }
        symbols[best_index] += symbols[best_index + 1];
        symbols.erase(symbols.begin() + best_index + 1);
    }

    std::vector<uint64_t> ids;
    ids.reserve(symbols.size());
    for (const auto& s : symbols) {
        auto it = token_to_id_.find(s);
        if (it != token_to_id_.end()) {
            ids.push_back(it->second);
        } else {
            ids.push_back(unkOr(unk_id_));
        }
    }
    return ids;
}

std::vector<uint64_t> Tokenizer::unigramEncode(const std::string& text) const {
    if (text.empty()) return {};
    const auto& vocab = token_bytes_.empty() ? tokens_ : token_bytes_;
    // Dynamic programming over byte positions.
    size_t n = text.size();
    std::vector<float> best_score(n + 1, -std::numeric_limits<float>::infinity());
    std::vector<uint64_t> best_id(n + 1, std::numeric_limits<uint64_t>::max());
    std::vector<size_t> best_prev(n + 1, n + 1);
    best_score[0] = 0.0f;

    for (size_t pos = 0; pos < n; ++pos) {
        if (best_score[pos] == -std::numeric_limits<float>::infinity()) continue;
        // Traverse trie-like via bucket lookup.
        uint8_t lead = static_cast<uint8_t>(text[pos]);
        auto it_bucket = bucket_.find(lead);
        if (it_bucket != bucket_.end()) {
            for (uint64_t cid : it_bucket->second) {
                const std::string& tok = vocab[cid];
                size_t len = tok.size();
                if (len == 0 || pos + len > n) continue;
                if (std::memcmp(text.data() + pos, tok.data(), len) != 0) continue;
                float score = best_score[pos];
                if (cid < scores_.size()) score += scores_[cid];
                size_t next = pos + len;
                if (score > best_score[next]) {
                    best_score[next] = score;
                    best_prev[next] = pos;
                    best_id[next] = cid;
                }
            }
        }
        // Unknown fallback (single byte).
        float score = best_score[pos] + (unk_id_ < scores_.size() ? scores_[unk_id_] : -1.0f);
        size_t next = pos + 1;
        uint64_t unk = unkOr(unk_id_);
        if (score > best_score[next]) {
            best_score[next] = score;
            best_prev[next] = pos;
            best_id[next] = unk;
        }
    }

    if (best_id[n] == std::numeric_limits<uint64_t>::max()) {
        return {};
    }
    std::vector<uint64_t> ids;
    for (size_t cur = n; cur > 0 && cur <= n; ) {
        ids.push_back(best_id[cur]);
        size_t prev = best_prev[cur];
        if (prev >= cur) break;
        cur = prev;
    }
    std::reverse(ids.begin(), ids.end());
    return ids;
}

std::string Tokenizer::decodeTokenString(const std::string& raw) {
    std::string out;
    out.reserve(raw.size());
    for (size_t i = 0; i < raw.size();) {
        if (raw[i] == '<' && i + 5 < raw.size() &&
            raw[i + 1] == '0' && raw[i + 2] == 'x' &&
            std::isxdigit(static_cast<unsigned char>(raw[i + 3])) &&
            std::isxdigit(static_cast<unsigned char>(raw[i + 4])) &&
            raw[i + 5] == '>') {
            std::string hex = raw.substr(i + 3, 2);
            char* end = nullptr;
            long val = std::strtol(hex.c_str(), &end, 16);
            if (end && *end == '\0') {
                out.push_back(static_cast<char>(val));
                i += 6;
                continue;
            }
        }
        out.push_back(raw[i]);
        ++i;
    }
    return out;
}

void Tokenizer::tryInitLlamaCpp(const frontend::GGUFLoader& loader) {
#ifdef MLC_ENABLE_LLAMA_TOKENIZER
    static std::once_flag backend_once;
    std::call_once(backend_once, []() {
        llama_backend_init();
    });
    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    params.use_mmap = true;
    params.progress_callback_user_data = nullptr;
    params.progress_callback = nullptr;
    llama_model_ = llama_model_load_from_file(loader.path().c_str(), params);
    if (!llama_model_) {
        use_llama_cpp_ = false;
        return;
    }
    llama_vocab_ = llama_model_get_vocab(llama_model_);
    use_llama_cpp_ = (llama_vocab_ != nullptr);
    if (use_llama_cpp_) {
        bos_id_ = llama_vocab_bos(llama_vocab_);
        eos_id_ = llama_vocab_eos(llama_vocab_);
    }
#else
    (void)loader;
    use_llama_cpp_ = false;
#endif
}

std::vector<uint64_t> Tokenizer::encodeWithLlama(const std::string& text, bool add_special) const {
#ifdef MLC_ENABLE_LLAMA_TOKENIZER
    std::vector<uint64_t> out;
    if (!llama_vocab_) return out;
    const bool parse_special = true;
    // First pass to get required size.
    int32_t needed = llama_tokenize(llama_vocab_, text.c_str(), static_cast<int32_t>(text.size()),
                                    nullptr, 0, add_special, parse_special);
    if (needed < 0) needed = -needed;
    if (needed == 0) return out;
    std::vector<llama_token> tmp(static_cast<size_t>(needed));
    int32_t written = llama_tokenize(llama_vocab_, text.c_str(), static_cast<int32_t>(text.size()),
                                     tmp.data(),
                                     static_cast<int32_t>(tmp.size()),
                                     add_special,
                                     parse_special);
    if (written < 0) written = -written;
    out.reserve(static_cast<size_t>(written));
    for (int i = 0; i < written && i < needed; ++i) {
        out.push_back(static_cast<uint64_t>(tmp[static_cast<size_t>(i)]));
    }
    return out;
#else
    (void)text;
    return {};
#endif
}

std::string Tokenizer::decodeWithLlama(const std::vector<uint64_t>& tokens) const {
#ifdef MLC_ENABLE_LLAMA_TOKENIZER
    if (!llama_vocab_ || tokens.empty()) return {};
    // Estimate output size; allow growth.
    std::vector<llama_token> toks(tokens.begin(), tokens.end());
    int32_t guess = static_cast<int32_t>(tokens.size() * 8 + 32);
    std::vector<char> buffer(static_cast<size_t>(guess));
    int32_t written = llama_detokenize(llama_vocab_,
                                       toks.data(),
                                       static_cast<int32_t>(toks.size()),
                                       buffer.data(),
                                       static_cast<int32_t>(buffer.size()),
                                       true,  // remove_special
                                       false  // unparse_special
    );
    if (written < 0) {
        int32_t needed = -written;
        buffer.resize(static_cast<size_t>(needed));
        written = llama_detokenize(llama_vocab_,
                                   toks.data(),
                                   static_cast<int32_t>(toks.size()),
                                   buffer.data(),
                                   static_cast<int32_t>(buffer.size()),
                                   true,
                                   false);
    }
    if (written <= 0) return {};
    return std::string(buffer.data(), static_cast<size_t>(written));
#else
    (void)tokens;
    return {};
#endif
}

} // namespace runtime
} // namespace mlc
