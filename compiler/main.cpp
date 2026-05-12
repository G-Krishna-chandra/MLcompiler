#include <iostream>
#include <iomanip>
#include <sstream>
#include <iomanip>
#include <cctype>
#include "frontends/frontend.hpp"
#include "frontends/gguf_loader.hpp"
#include "frontends/gguf_to_ir.hpp"
#include "frontends/gguf_utils.hpp"
#include "ir/ir.hpp"
#include "runtime/session.hpp"
#include "runtime/execution_plan_builder.hpp"
#include "runtime/model_runner.hpp"
#include "runtime/metal_runtime.hpp"
#include "runtime/decode_runner.hpp"
#include "runtime/tokenizer.hpp"
#include "runtime/sampling.hpp"
#include "runtime/parity_harness.hpp"
#include "runtime/quantization.hpp"
#include "runtime/execution_executor.hpp"
#include "runtime/execution_context.hpp"
#include "runtime/operator_backend.hpp"
#include "runtime/kernel_registry.hpp"
#include "util/cli_helpers.hpp"

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

#ifdef MLC_ENABLE_LLAMA_TOKENIZER
#include "llama.h"
#endif

#include <random>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <fstream>

namespace {
    std::string trim(const std::string& s) {
        size_t start = 0;
        while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
            ++start;
        }
        size_t end = s.size();
        while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
            --end;
        }
        return s.substr(start, end - start);
    }

    std::vector<float> parseFloatList(const std::string& text) {
        std::vector<float> values;
        std::string normalized = text;
        for (char& c : normalized) {
            if (c == ',') c = ' ';
        }
        std::stringstream ss(normalized);
        float v = 0.0f;
        while (ss >> v) {
            values.push_back(v);
        }
        if (values.empty()) {
            throw std::runtime_error("Input list must contain at least one value");
        }
        return values;
    }

    struct ChatMessage {
        std::string role;
        std::string content;
    };

    bool templateHasBosHint(const std::string& tmpl) {
        if (tmpl.find("bos_token") != std::string::npos) return true;
        if (tmpl.find("<s>") != std::string::npos) return true;
        if (tmpl.find("<|begin_of_text|>") != std::string::npos) return true;
        return false;
    }

    std::string renderChatTemplateSimple(const std::string& tmpl,
                                         const std::vector<ChatMessage>& messages,
                                         bool add_generation_prompt,
                                         const mlc::runtime::Tokenizer& tokenizer) {
        if (tmpl.find("<|user|>") == std::string::npos &&
            tmpl.find("<|assistant|>") == std::string::npos &&
            tmpl.find("<|system|>") == std::string::npos) {
            return {};
        }
        std::string eos = tokenizer.tokenString(tokenizer.eosId());
        if (eos.empty()) {
            eos = "</s>";
        }
        std::string bos;
        if (templateHasBosHint(tmpl)) {
            bos = tokenizer.tokenString(tokenizer.bosId());
            if (bos.empty()) {
                bos = "<s>";
            }
        }
        std::string out;
        if (!bos.empty()) {
            out += bos;
        }
        for (const auto& msg : messages) {
            if (msg.role == "user") {
                out += "<|user|>\n";
            } else if (msg.role == "assistant") {
                out += "<|assistant|>\n";
            } else if (msg.role == "system") {
                out += "<|system|>\n";
            } else {
                continue;
            }
            out += msg.content;
            out += eos;
        }
        if (add_generation_prompt) {
            out += "<|assistant|>";
        }
        return out;
    }

#ifdef MLC_ENABLE_LLAMA_TOKENIZER
    std::string applyChatTemplateLlama(const std::string& tmpl,
                                       const std::vector<ChatMessage>& messages,
                                       bool add_generation_prompt,
                                       std::string* error) {
        if (tmpl.empty()) {
            if (error) *error = "empty template";
            return {};
        }

        std::vector<std::string> roles;
        std::vector<std::string> contents;
        std::vector<llama_chat_message> chat;
        roles.reserve(messages.size());
        contents.reserve(messages.size());
        chat.reserve(messages.size());
        for (const auto& msg : messages) {
            roles.push_back(msg.role);
            contents.push_back(msg.content);
            chat.push_back({roles.back().c_str(), contents.back().c_str()});
        }

        int32_t needed = llama_chat_apply_template(tmpl.c_str(),
                                                   chat.data(),
                                                   chat.size(),
                                                   add_generation_prompt,
                                                   nullptr,
                                                   0);
        if (needed < 0) {
            if (error) *error = "template apply failed (size)";
            return {};
        }
        std::string out;
        out.resize(static_cast<size_t>(needed));
        int32_t written = llama_chat_apply_template(tmpl.c_str(),
                                                    chat.data(),
                                                    chat.size(),
                                                    add_generation_prompt,
                                                    out.data(),
                                                    static_cast<int32_t>(out.size()));
        if (written < 0) {
            if (error) *error = "template apply failed (render)";
            return {};
        }
        if (static_cast<size_t>(written) < out.size()) {
            out.resize(static_cast<size_t>(written));
        }
        return out;
    }
#endif

    uint64_t calculateElementCount(const std::vector<int64_t>& shape) {
        uint64_t count = 1;
        for (int64_t dim : shape) {
            count *= static_cast<uint64_t>(dim);
        }
        return count;
    }
    
    uint64_t calculateByteSize(const mlc::ir::Tensor* tensor) {
        uint64_t element_count = calculateElementCount(tensor->shape);
        // Approximate size based on dtype
        switch (tensor->dtype) {
            case mlc::ir::DataType::F32: return element_count * 4;
            case mlc::ir::DataType::F16: return element_count * 2;
            case mlc::ir::DataType::BF16: return element_count * 2;
            case mlc::ir::DataType::I8: return element_count * 1;
            case mlc::ir::DataType::I4: return (element_count + 1) / 2;  // Packed
            default: return element_count * 4;
        }
    }
    
    bool parseUnsigned(const std::string& text, uint64_t& out) {
        size_t idx = 0;
        try {
            out = std::stoull(text, &idx);
        } catch (...) {
            return false;
        }
        return idx == text.size();
    }

    bool parseSizeT(const std::string& text, size_t& out) {
        uint64_t tmp = 0;
        if (!parseUnsigned(text, tmp)) return false;
        out = static_cast<size_t>(tmp);
        return true;
    }

    std::vector<uint64_t> parseTokenList(const std::string& text) {
        std::vector<uint64_t> tokens;
        std::stringstream ss(text);
        std::string item;
        while (std::getline(ss, item, ',')) {
            item = trim(item);
            if (item.empty()) continue;
            uint64_t value = 0;
            if (!parseUnsigned(item, value)) {
                throw std::runtime_error("Invalid token id: " + item);
            }
            tokens.push_back(value);
        }
        if (tokens.empty()) {
            throw std::runtime_error("Token list must contain at least one id");
        }
        return tokens;
    }

    struct LogitsTraceConfig {
        FILE* file = nullptr;
        size_t top_k_override = 0;
    };

    struct EmbeddingsTraceConfig {
        FILE* file = nullptr;
        size_t preview = 0;
    };

    struct ChatTraceConfig {
        FILE* prompt_file = nullptr;
        FILE* tokens_file = nullptr;
    };

    bool parseSizeTEnv(const char* text, size_t& out) {
        if (!text || !*text) return false;
        char* end = nullptr;
        unsigned long value = std::strtoul(text, &end, 10);
        if (!end || *end != '\0') return false;
        out = static_cast<size_t>(value);
        return true;
    }

    const LogitsTraceConfig& logitsTraceConfig() {
        static LogitsTraceConfig cfg = []() {
            LogitsTraceConfig out;
            const char* path = std::getenv("MLC_LOGITS_FILE");
            if (path && *path) {
                out.file = std::fopen(path, "a");
                if (out.file) {
                    std::setvbuf(out.file, nullptr, _IOLBF, 0);
                }
            }
            size_t top_k = 0;
            const char* env_topk = std::getenv("MLC_LOGITS_TOPK");
            if (parseSizeTEnv(env_topk, top_k)) {
                out.top_k_override = top_k;
            }
            return out;
        }();
        return cfg;
    }

    const EmbeddingsTraceConfig& embeddingsTraceConfig() {
        static EmbeddingsTraceConfig cfg = []() {
            EmbeddingsTraceConfig out;
            const char* path = std::getenv("MLC_EMBEDDINGS_FILE");
            if (path && *path) {
                out.file = std::fopen(path, "a");
                if (out.file) {
                    std::setvbuf(out.file, nullptr, _IOLBF, 0);
                }
            }
            size_t preview = 0;
            const char* env_preview = std::getenv("MLC_EMBEDDINGS_PREVIEW");
            if (parseSizeTEnv(env_preview, preview)) {
                out.preview = preview;
            } else {
                out.preview = 16;
            }
            return out;
        }();
        return cfg;
    }

    const ChatTraceConfig& chatTraceConfig() {
        static ChatTraceConfig cfg = []() {
            ChatTraceConfig out;
            const char* prompt_path = std::getenv("MLC_CHAT_PROMPT_FILE");
            if (prompt_path && *prompt_path) {
                out.prompt_file = std::fopen(prompt_path, "a");
                if (out.prompt_file) {
                    std::setvbuf(out.prompt_file, nullptr, _IOLBF, 0);
                }
            }
            const char* tokens_path = std::getenv("MLC_CHAT_TOKENS_FILE");
            if (tokens_path && *tokens_path) {
                out.tokens_file = std::fopen(tokens_path, "a");
                if (out.tokens_file) {
                    std::setvbuf(out.tokens_file, nullptr, _IOLBF, 0);
                }
            }
            return out;
        }();
        return cfg;
    }

    void logChatPrompt(const ChatTraceConfig& cfg,
                       size_t turn,
                       bool used_template,
                       const std::string& rendered) {
        if (!cfg.prompt_file) return;
        std::fprintf(cfg.prompt_file,
                     "[ChatPrompt] turn=%zu used_template=%d length=%zu\n",
                     turn,
                     used_template ? 1 : 0,
                     rendered.size());
        if (!rendered.empty()) {
            std::fwrite(rendered.data(), 1, rendered.size(), cfg.prompt_file);
            if (rendered.back() != '\n') {
                std::fputc('\n', cfg.prompt_file);
            }
        }
        std::fprintf(cfg.prompt_file, "[ChatPromptEnd]\n");
        std::fflush(cfg.prompt_file);
    }

    size_t resolveTopK(const LogitsTraceConfig& cfg, size_t default_top_k) {
        if (cfg.top_k_override > 0) return cfg.top_k_override;
        if (default_top_k > 0) return default_top_k;
        return 5;
    }

    std::vector<std::pair<uint64_t, float>>
    topKLogits(const std::vector<float>& logits, size_t top_k) {
        std::vector<std::pair<uint64_t, float>> out;
        if (logits.empty() || top_k == 0) return out;
        top_k = std::min(top_k, logits.size());
        std::vector<std::pair<float, uint64_t>> scored;
        scored.reserve(logits.size());
        for (uint64_t i = 0; i < logits.size(); ++i) {
            scored.push_back({logits[i], i});
        }
        if (top_k < scored.size()) {
            std::nth_element(scored.begin(),
                             scored.begin() + top_k,
                             scored.end(),
                             [](const auto& a, const auto& b) {
                                 return a.first > b.first;
                             });
            scored.resize(top_k);
        }
        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        out.reserve(top_k);
        for (const auto& p : scored) {
            out.push_back({p.second, p.first});
        }
        return out;
    }

    void logTokenListToFile(FILE* file,
                            const char* label,
                            size_t seq,
                            const std::vector<uint64_t>& tokens) {
        if (!file) return;
        std::fprintf(file, "[Tokens] seq=%zu label=%s count=%zu ids=",
                     seq,
                     label ? label : "tokens",
                     tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i > 0) std::fprintf(file, ",");
            std::fprintf(file, "%llu",
                         static_cast<unsigned long long>(tokens[i]));
        }
        std::fprintf(file, "\n");
        std::fflush(file);
    }

    void logTokenList(const LogitsTraceConfig& cfg,
                      const char* label,
                      size_t seq,
                      const std::vector<uint64_t>& tokens) {
        logTokenListToFile(cfg.file, label, seq, tokens);
    }

    void logLogits(const LogitsTraceConfig& cfg,
                   const char* phase,
                   size_t seq,
                   size_t step,
                   uint64_t token,
                   size_t pos,
                   const std::vector<float>& logits,
                   size_t default_top_k) {
        if (!cfg.file || logits.empty()) return;
        size_t top_k = resolveTopK(cfg, default_top_k);
        if (top_k == 0) return;
        float min_val = logits.front();
        float max_val = logits.front();
        double sum = 0.0;
        for (float v : logits) {
            min_val = std::min(min_val, v);
            max_val = std::max(max_val, v);
            sum += static_cast<double>(v);
        }
        double mean = sum / static_cast<double>(logits.size());
        auto top = topKLogits(logits, top_k);
        std::fprintf(cfg.file,
                     "[Logits] seq=%zu phase=%s step=%zu token=%llu pos=%zu count=%zu min=%.6g max=%.6g mean=%.6g topk=%zu entries=",
                     seq,
                     phase ? phase : "step",
                     step,
                     static_cast<unsigned long long>(token),
                     pos,
                     logits.size(),
                     min_val,
                     max_val,
                     mean,
                     top.size());
        for (size_t i = 0; i < top.size(); ++i) {
            if (i > 0) std::fprintf(cfg.file, ",");
            std::fprintf(cfg.file,
                         "%llu:%.6g",
                         static_cast<unsigned long long>(top[i].first),
                         top[i].second);
        }
        std::fprintf(cfg.file, "\n");
        std::fflush(cfg.file);
    }

    void logEmbeddings(const EmbeddingsTraceConfig& cfg,
                       const char* phase,
                       size_t seq,
                       size_t step,
                       uint64_t token,
                       size_t pos,
                       const float* data,
                       size_t count) {
        if (!cfg.file || !data || count == 0) return;
        size_t preview = std::min(cfg.preview, count);
        if (preview == 0) return;
        float min_val = data[0];
        float max_val = data[0];
        double sum = 0.0;
        for (size_t i = 0; i < count; ++i) {
            min_val = std::min(min_val, data[i]);
            max_val = std::max(max_val, data[i]);
            sum += static_cast<double>(data[i]);
        }
        double mean = sum / static_cast<double>(count);
        std::fprintf(cfg.file,
                     "[Embeddings] seq=%zu phase=%s step=%zu token=%llu pos=%zu count=%zu min=%.6g max=%.6g mean=%.6g values=",
                     seq,
                     phase ? phase : "step",
                     step,
                     static_cast<unsigned long long>(token),
                     pos,
                     count,
                     min_val,
                     max_val,
                     mean);
        for (size_t i = 0; i < preview; ++i) {
            if (i > 0) std::fprintf(cfg.file, ",");
            std::fprintf(cfg.file, "%.6g", data[i]);
        }
        std::fprintf(cfg.file, "\n");
        std::fflush(cfg.file);
    }

    std::vector<std::vector<uint64_t>> parseTokenBatches(const std::string& text) {
        std::vector<std::vector<uint64_t>> batches;
        std::stringstream ss(text);
        std::string segment;
        while (std::getline(ss, segment, ';')) {
            std::string cleaned = trim(segment);
            if (cleaned.empty()) {
                continue;
            }
            batches.push_back(parseTokenList(cleaned));
        }
        if (batches.empty()) {
            throw std::runtime_error("Token list must contain at least one sequence");
        }
        return batches;
    }

    // Lightweight shell escaping for single-quoted strings.
    std::string shellEscape(const std::string& text) {
        std::string out;
        out.reserve(text.size() + 2);
        out.push_back('\'');
        for (char c : text) {
            if (c == '\'') {
                out.append("'\"'\"'");
            } else {
                out.push_back(c);
            }
        }
        out.push_back('\'');
        return out;
    }

    // Replace all occurrences of placeholder with escaped value.
    std::string substitute(const std::string& templ,
                           const std::string& placeholder,
                           const std::string& value) {
        std::string out;
        size_t pos = 0;
        while (true) {
            size_t found = templ.find(placeholder, pos);
            if (found == std::string::npos) {
                out.append(templ.substr(pos));
                break;
            }
            out.append(templ.substr(pos, found - pos));
            out.append(value);
            pos = found + placeholder.size();
        }
        return out;
    }

    std::string runCommandCapture(const std::string& cmd) {
        std::array<char, 256> buffer{};
        std::string result;
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) {
            throw std::runtime_error("Failed to run command: " + cmd);
        }
        while (fgets(buffer.data(), buffer.size(), pipe)) {
            result.append(buffer.data());
        }
        int rc = pclose(pipe);
        if (rc != 0) {
            throw std::runtime_error("Tokenizer command failed with exit code " + std::to_string(rc));
        }
        return result;
    }

    std::vector<uint64_t> parseIdsFromOutput(const std::string& text) {
        std::vector<uint64_t> ids;
        // If output contains a bracketed list (e.g., [1, 2, 3]), focus on that substring.
        size_t l = text.rfind('[');
        size_t r = text.rfind(']');
        std::string view = text;
        if (l != std::string::npos && r != std::string::npos && r > l) {
            view = text.substr(l + 1, r - l - 1);
        }
        std::string current;
        auto flush = [&]() {
            if (current.empty()) return;
            uint64_t v = 0;
            if (parseUnsigned(current, v)) {
                ids.push_back(v);
            }
            current.clear();
        };
        for (char c : view) {
            if (std::isdigit(static_cast<unsigned char>(c))) {
                current.push_back(c);
            } else {
                flush();
            }
        }
        flush();
        return ids;
    }

    std::vector<uint64_t> tokenizeExternal(const std::string& cmd_template,
                                           const std::string& text) {
        if (cmd_template.empty()) return {};
        std::string escaped = shellEscape(text);
        std::string cmd = substitute(cmd_template, "{text}", escaped);
        std::string out = runCommandCapture(cmd);
        return parseIdsFromOutput(out);
    }

    std::string detokenizeExternal(const std::string& cmd_template,
                                   const std::vector<uint64_t>& ids) {
        if (cmd_template.empty() || ids.empty()) return {};
        std::ostringstream oss;
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i > 0) oss << ",";
            oss << ids[i];
        }
        std::string escaped_ids = shellEscape(oss.str());
        std::string cmd = substitute(cmd_template, "{ids}", escaped_ids);
        return runCommandCapture(cmd);
    }

    void printDecodeResult(const mlc::runtime::DecodeResult& res, size_t preview) {
        if (!res.success) {
            std::cout << "Decode failed after " << res.steps.size() << " steps.\n";
        } else {
            std::cout << "Decode completed " << res.steps.size() << " steps.\n";
        }
        for (size_t i = 0; i < res.steps.size(); ++i) {
            const auto& step = res.steps[i];
            std::cout << "Step " << i << " token=" << step.token
                      << " pos=" << step.position
                      << " success=" << (step.success ? "yes" : "no") << "\n";
            if (!step.trace.empty()) {
                std::cout << "  trace:\n";
                for (const auto& line : step.trace) {
                    std::cout << "    - " << line << "\n";
                }
            }
            if (!step.success && !step.error.empty()) {
                std::cout << "  error: " << step.error << "\n";
            }
            if (!step.top_indices.empty()) {
                std::cout << "  topk: ";
                for (size_t j = 0; j < step.top_indices.size(); ++j) {
                    if (j > 0) std::cout << ", ";
                    std::cout << step.top_indices[j] << " (" << step.top_probs[j] << ")";
                }
                std::cout << "\n";
            }
            if (!step.logits.empty()) {
                std::cout << "  logits: ";
                size_t n = std::min(preview, step.logits.size());
                for (size_t j = 0; j < n; ++j) {
                    if (j > 0) std::cout << ", ";
                    std::cout << step.logits[j];
                }
                if (n < step.logits.size()) std::cout << ", ...";
                std::cout << "\n";
            }
        }
    }

    void printFloatPreview(const std::vector<float>& values) {
        if (values.empty()) {
            std::cout << "(empty)\n";
            return;
        }
        for (size_t i = 0; i < values.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << values[i];
        }
        std::cout << "\n";
    }

    std::string ggufValueToString(const mlc::frontend::GGUFValue& value) {
        std::ostringstream oss;
        switch (value.type) {
            case mlc::frontend::GGUFValueType::UINT8:
                oss << static_cast<uint32_t>(std::get<uint8_t>(value.data));
                break;
            case mlc::frontend::GGUFValueType::INT8:
                oss << static_cast<int32_t>(std::get<int8_t>(value.data));
                break;
            case mlc::frontend::GGUFValueType::UINT16:
                oss << std::get<uint16_t>(value.data);
                break;
            case mlc::frontend::GGUFValueType::INT16:
                oss << std::get<int16_t>(value.data);
                break;
            case mlc::frontend::GGUFValueType::UINT32:
                oss << std::get<uint32_t>(value.data);
                break;
            case mlc::frontend::GGUFValueType::INT32:
                oss << std::get<int32_t>(value.data);
                break;
            case mlc::frontend::GGUFValueType::UINT64:
                oss << std::get<uint64_t>(value.data);
                break;
            case mlc::frontend::GGUFValueType::INT64:
                oss << std::get<int64_t>(value.data);
                break;
            case mlc::frontend::GGUFValueType::FLOAT32:
                oss << std::fixed << std::setprecision(6) << std::get<float>(value.data);
                break;
            case mlc::frontend::GGUFValueType::FLOAT64:
                oss << std::fixed << std::setprecision(6) << std::get<double>(value.data);
                break;
            case mlc::frontend::GGUFValueType::BOOL:
                oss << (std::get<bool>(value.data) ? "true" : "false");
                break;
            case mlc::frontend::GGUFValueType::STRING:
                oss << "\"" << std::get<std::string>(value.data) << "\"";
                break;
        case mlc::frontend::GGUFValueType::ARRAY: {
            const auto& arr = std::get<std::vector<mlc::frontend::GGUFValue>>(value.data);
            oss << "[";
            for (size_t i = 0; i < arr.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << ggufValueToString(arr[i]);
            }
            oss << "]";
            break;
        }
        case mlc::frontend::GGUFValueType::RAW: {
            const auto& bytes = std::get<std::vector<uint8_t>>(value.data);
            oss << "<raw " << bytes.size() << " bytes>";
            break;
        }
        default:
            oss << "<unknown type>";
    }
        return oss.str();
    }
    
    void printGGUFHeader(const mlc::frontend::GGUFLoader& loader) {
        const auto& header = loader.header();
        std::cout << "GGUF Header:\n";
        std::cout << "  Magic: 0x" << std::hex << std::setw(8) << std::setfill('0') << header.magic << std::dec << "\n";
        std::cout << "  Version: " << header.version << "\n";
        std::cout << "  Tensors: " << header.n_tensors << "\n";
        std::cout << "  KV pairs: " << header.n_kv << "\n\n";
        
        const auto& kv_metadata = loader.kvMetadata();
        if (!kv_metadata.empty()) {
            std::cout << "KV Metadata (" << kv_metadata.size() << " entries):\n";
            for (const auto& [key, value] : kv_metadata) {
                std::cout << "  " << std::left << std::setw(40) << key 
                          << " = " << ggufValueToString(value) << "\n";
            }
        } else {
            std::cout << "KV Metadata: (none)\n";
        }
        std::cout << "\n";
    }
    
    void printTensorDetails(const mlc::ir::Tensor* tensor) {
        std::cout << "Tensor: " << tensor->name << "\n";
        std::cout << "  dtype: " << mlc::ir::dataTypeToString(tensor->dtype);
        if (tensor->metadata.find("gguf_dtype") != tensor->metadata.end()) {
            std::cout << " (gguf_dtype: " << tensor->metadata.at("gguf_dtype") << ")";
        }
        std::cout << "\n";
        
        std::cout << "  shape: [";
        for (size_t i = 0; i < tensor->shape.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << tensor->shape[i];
        }
        std::cout << "]\n";
        
        if (!tensor->original_shape.empty() && tensor->original_shape != tensor->shape) {
            std::cout << "  original_shape: [";
            for (size_t i = 0; i < tensor->original_shape.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << tensor->original_shape[i];
            }
            std::cout << "]\n";
        }
        
        std::cout << "  byte_offset: " << tensor->byteOffset << "\n";
        
        uint64_t elem_count = calculateElementCount(tensor->shape);
        uint64_t byte_size = calculateByteSize(tensor);
        std::cout << "  element_count: " << elem_count << "\n";
        std::cout << "  expected_byte_size: " << byte_size << "\n";
        std::cout << "  layout_transposed: " << (tensor->layout_transposed ? "true" : "false") << "\n";
        
        if (!tensor->metadata.empty()) {
            std::cout << "  metadata:\n";
            for (const auto& [key, value] : tensor->metadata) {
                std::cout << "    " << key << " = " << value << "\n";
            }
        }
        std::cout << "\n";
    }
    
    void printTensorHex(const mlc::frontend::GGUFLoader& loader, 
                        const mlc::ir::Tensor* tensor, int num_bytes) {
        // Find the tensor in GGUF loader
        const auto& gguf_tensors = loader.tensors();
        auto it = gguf_tensors.find(tensor->name);
        if (it == gguf_tensors.end()) {
            std::cerr << "Error: Tensor not found in GGUF loader\n";
            return;
        }
        
        auto data = loader.loadTensorData(it->second);
        int bytes_to_print = std::min(num_bytes, static_cast<int>(data.size()));
        
        std::cout << "Hex dump of first " << bytes_to_print << " bytes:\n";
        for (int i = 0; i < bytes_to_print; ++i) {
            if (i > 0 && i % 16 == 0) std::cout << "\n";
            std::cout << std::hex << std::setw(2) << std::setfill('0') 
                      << static_cast<int>(data[i]) << " ";
        }
        std::cout << std::dec << "\n\n";
    }
    
    void printGGUFMeta(const mlc::frontend::GGUFLoader& loader) {
        const auto& kv_metadata = loader.kvMetadata();
        
        // Extract common LLaMA/transformer metadata
        std::cout << "GGUF Metadata:\n";
        std::cout << "===============\n\n";
        
        // Architecture
        auto arch_it = kv_metadata.find("general.architecture");
        if (arch_it != kv_metadata.end()) {
            if (arch_it->second.type == mlc::frontend::GGUFValueType::STRING) {
                std::cout << "arch: " << std::get<std::string>(arch_it->second.data) << "\n";
            }
        }
        
        // Vocabulary size
        auto n_vocab_it = kv_metadata.find("tokenizer.ggml.vocab_size");
        if (n_vocab_it == kv_metadata.end()) {
            n_vocab_it = kv_metadata.find("tokenizer.vocab_size");
        }
        if (n_vocab_it != kv_metadata.end()) {
            if (n_vocab_it->second.type == mlc::frontend::GGUFValueType::UINT32) {
                std::cout << "n_vocab: " << std::get<uint32_t>(n_vocab_it->second.data) << "\n";
            } else if (n_vocab_it->second.type == mlc::frontend::GGUFValueType::UINT64) {
                std::cout << "n_vocab: " << std::get<uint64_t>(n_vocab_it->second.data) << "\n";
            }
        }
        
        // Embedding dimension
        auto n_embd_it = kv_metadata.find(std::string("llama.embedding_length"));
        if (n_embd_it == kv_metadata.end()) {
            n_embd_it = kv_metadata.find("transformer.embedding_length");
        }
        if (n_embd_it == kv_metadata.end()) {
            n_embd_it = kv_metadata.find("model.embedding_length");
        }
        if (n_embd_it != kv_metadata.end()) {
            if (n_embd_it->second.type == mlc::frontend::GGUFValueType::UINT32) {
                std::cout << "n_embd: " << std::get<uint32_t>(n_embd_it->second.data) << "\n";
            } else if (n_embd_it->second.type == mlc::frontend::GGUFValueType::UINT64) {
                std::cout << "n_embd: " << std::get<uint64_t>(n_embd_it->second.data) << "\n";
            }
        }
        
        // Print other common fields
        for (const auto& [key, value] : kv_metadata) {
            if (key.find("general.") == 0 || key.find("llama.") == 0 || 
                key.find("transformer.") == 0 || key.find("model.") == 0) {
                if (key != "general.architecture") {  // Already printed
                    std::cout << key << ": " << ggufValueToString(value) << "\n";
                }
            }
        }
        
        std::cout << "\nTensors:\n";
        std::cout << "========\n";
        const auto& tensors = loader.tensors();
        for (const auto& [name, tensor_info] : tensors) {
            std::cout << name << ": shape=[";
            for (size_t i = 0; i < tensor_info.shape.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << tensor_info.shape[i];
            }
            std::cout << "] dtype=" << mlc::frontend::ggufDtypeToString(tensor_info.dtype) << "\n";
        }
        std::cout << "\n";
    }
    
    int handleMetaCommand(const std::vector<std::string>& args) {
        if (args.empty()) {
            std::cerr << "Error: meta command requires a GGUF file path\n";
            std::cerr << "Usage: mlc meta <gguf_path>\n";
            return 1;
        }
        
        const std::string& gguf_path = args[0];
        
        try {
            mlc::frontend::GGUFLoader loader(gguf_path);
            if (!loader.load()) {
                std::cerr << "Error: Failed to load GGUF file: " << gguf_path << "\n";
                return 1;
            }
            
            printGGUFMeta(loader);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
            return 1;
        }
        
        return 0;
    }
    
    int handleInspectCommand(const mlc::util::ParsedArgs& args) {
        if (args.arguments.empty()) {
            std::cerr << "Error: inspect command requires a GGUF file path\n";
            std::cerr << "Usage: mlc inspect [options] <gguf_path>\n";
            return 1;
        }
        
        const std::string& gguf_path = args.arguments[0];
        
        try {
            // Load GGUF file
            mlc::frontend::GGUFLoader loader(gguf_path);
            loader.setVerbose(args.verbose);
            if (!loader.load()) {
                std::cerr << "Error: Failed to load GGUF file: " << gguf_path << "\n";
                return 1;
            }
            
            // Convert to IR
            auto graph = mlc::frontend::GGUFToIR(loader);
            
            // Print header
            std::cout << "GGUF Model: " << gguf_path << "\n";
            std::cout << "========================================\n\n";
            
            // Handle --dump-header
            if (args.dump_header) {
                printGGUFHeader(loader);
            }
            
            // Handle --dump-one
            if (!args.dump_one.empty()) {
                const auto& tensors = graph->tensors();
                auto it = std::find_if(tensors.begin(), tensors.end(),
                    [&](const mlc::ir::Tensor* t) { return t->name == args.dump_one; });
                
                if (it != tensors.end()) {
                    printTensorDetails(*it);
                    if (args.hex_bytes > 0) {
                        printTensorHex(loader, *it, args.hex_bytes);
                    }
                } else {
                    std::cerr << "Error: Tensor '" << args.dump_one << "' not found\n";
                    return 1;
                }
                return 0;
            }
            
            // Handle --dump-tensors
            if (args.dump_tensors) {
                const auto& tensors = graph->tensors();
                for (const auto* tensor : tensors) {
                    printTensorDetails(tensor);
                }
                std::cout << "Nodes: " << graph->nodes().size() << "\n";
                return 0;
            }
            
            // Default: print tensor list
            const auto& tensors = graph->tensors();
            std::cout << "Tensors (" << tensors.size() << "):\n\n";
            std::cout << std::left << std::setw(40) << "Name" 
                      << std::setw(25) << "Shape" 
                      << std::setw(10) << "Dtype" 
                      << "Offset" << "\n";
            std::cout << std::string(85, '-') << "\n";
            
            for (const auto* tensor : tensors) {
                std::cout << std::left << std::setw(40) << tensor->name;
                
                std::string shape_str = "[";
                for (size_t i = 0; i < tensor->shape.size(); ++i) {
                    if (i > 0) shape_str += ", ";
                    shape_str += std::to_string(tensor->shape[i]);
                }
                shape_str += "]";
                std::cout << std::setw(25) << shape_str;
                std::cout << std::setw(10) << mlc::ir::dataTypeToString(tensor->dtype);
                std::cout << tensor->byteOffset << "\n";
            }
            
            std::cout << "\nNodes: " << graph->nodes().size() << "\n";
            
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
            return 1;
        }
        
        return 0;
    }

    int handleLinearCommand(const std::vector<std::string>& args) {
        if (args.size() < 3) {
            std::cerr << "Error: linear command requires <gguf_path> <tensor_name> <comma-separated-input>\n";
            return 1;
        }

        const std::string& gguf_path = args[0];
        const std::string& tensor_name = args[1];
        std::string input_values = args[2];
        size_t top_k = 0;
        std::string output_path;
        for (size_t i = 3; i < args.size(); ++i) {
            const std::string& arg = args[i];
            if (arg == "--topk" && i + 1 < args.size()) {
                if (!parseSizeT(args[++i], top_k)) {
                    std::cerr << "Error: invalid value for --topk\n";
                    return 1;
                }
            } else if (arg.rfind("--topk=", 0) == 0) {
                if (!parseSizeT(arg.substr(7), top_k)) {
                    std::cerr << "Error: invalid value for --topk\n";
                    return 1;
                }
            } else if (arg == "--output-file" && i + 1 < args.size()) {
                output_path = args[++i];
            } else if (arg.rfind("--output-file=", 0) == 0) {
                output_path = arg.substr(14);
            } else {
                std::cerr << "Unknown linear option: " << arg << "\n";
                return 1;
            }
        }

        try {
            if (!input_values.empty() && input_values[0] == '@') {
                const std::string path = input_values.substr(1);
                std::ifstream file(path);
                if (!file) {
                    std::cerr << "Error: failed to read input file " << path << "\n";
                    return 1;
                }
                std::ostringstream oss;
                oss << file.rdbuf();
                input_values = oss.str();
            }
            auto inputs = parseFloatList(input_values);
            mlc::runtime::Session session(gguf_path);
            auto outputs = session.runLinear(tensor_name, inputs);

            std::ostream* out = &std::cout;
            std::ofstream file_out;
            if (!output_path.empty()) {
                file_out.open(output_path);
                if (!file_out) {
                    std::cerr << "Error: failed to open output file " << output_path << "\n";
                    return 1;
                }
                out = &file_out;
            }
            if (top_k > 0) {
                if (top_k > outputs.size()) top_k = outputs.size();
                std::vector<std::pair<float, size_t>> scored;
                scored.reserve(outputs.size());
                for (size_t i = 0; i < outputs.size(); ++i) {
                    scored.emplace_back(outputs[i], i);
                }
                std::partial_sort(scored.begin(),
                                  scored.begin() + top_k,
                                  scored.end(),
                                  [](const auto& a, const auto& b) { return a.first > b.first; });
                scored.resize(top_k);
                *out << "Top-" << top_k << " logits:\n";
                for (size_t i = 0; i < scored.size(); ++i) {
                    *out << scored[i].second << ":" << scored[i].first;
                    if (i + 1 < scored.size()) *out << ", ";
                }
                *out << "\n";
            } else {
                *out << "Linear output (" << outputs.size() << " values):\n";
                for (size_t i = 0; i < outputs.size(); ++i) {
                    if (i > 0) *out << ", ";
                    *out << outputs[i];
                }
                *out << "\n";
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
            return 1;
        }

        return 0;
    }

int handleEmbedCommand(const std::vector<std::string>& args) {
    if (args.size() < 3) {
        std::cerr << "Error: embed command requires <gguf_path> <tensor_name> <token_id>\n";
        return 1;
    }

    const std::string& gguf_path = args[0];
    const std::string& tensor_name = args[1];
    uint64_t token_id = 0;
    try {
        token_id = std::stoull(args[2]);
    } catch (...) {
        std::cerr << "Error: token_id must be an integer\n";
        return 1;
    }

    try {
        mlc::runtime::Session session(gguf_path);
        auto embedding = session.getEmbedding(tensor_name, token_id);

        std::cout << "Embedding (" << embedding.size() << " values):\n";
        for (size_t i = 0; i < embedding.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << embedding[i];
        }
        std::cout << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

int handlePlanCommand(const std::vector<std::string>& args) {
    if (args.empty()) {
        std::cerr << "Error: plan command requires a GGUF file path\n";
        std::cerr << "Usage: mlc plan <gguf_path>\n";
        return 1;
    }

    const std::string& gguf_path = args[0];
    try {
        mlc::frontend::GGUFLoader loader(gguf_path);
        loader.load();
        auto plan = mlc::runtime::ExecutionPlanBuilder::BuildFromLoader(loader);
        std::cout << "Execution Plan (" << plan.nodes().size() << " nodes)\n";
        std::cout << plan.dump() << "\n";
        auto order = plan.topologicalOrder();
        std::cout << "Schedule: ";
        for (size_t i = 0; i < order.size(); ++i) {
            if (i > 0) std::cout << " -> ";
            std::cout << order[i];
        }
        std::cout << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

int handleRunCommand(const mlc::util::ParsedArgs& parsed) {
    std::vector<std::string> positional;
    uint64_t token_id = 0;
    size_t preview_len = 8;
    bool try_logits = true;
    bool simulate_plan = false;
    bool execute_plan = false;
    size_t simulate_limit = 0;
    size_t sequence_position = 0;

    for (size_t i = 0; i < parsed.arguments.size(); ++i) {
        const std::string& arg = parsed.arguments[i];
        if (arg == "--token" && i + 1 < parsed.arguments.size()) {
            if (!parseUnsigned(parsed.arguments[++i], token_id)) {
                std::cerr << "Error: invalid value for --token\n";
                return 1;
            }
        } else if (arg.rfind("--token=", 0) == 0) {
            if (!parseUnsigned(arg.substr(8), token_id)) {
                std::cerr << "Error: invalid value for --token\n";
                return 1;
            }
        } else if (arg == "--preview" && i + 1 < parsed.arguments.size()) {
            if (!parseSizeT(parsed.arguments[++i], preview_len)) {
                std::cerr << "Error: invalid value for --preview\n";
                return 1;
            }
        } else if (arg.rfind("--preview=", 0) == 0) {
            if (!parseSizeT(arg.substr(10), preview_len)) {
                std::cerr << "Error: invalid value for --preview\n";
                return 1;
            }
        } else if (arg == "--no-logits") {
            try_logits = false;
        } else if (arg == "--simulate") {
            simulate_plan = true;
        } else if (arg == "--simulate-limit" && i + 1 < parsed.arguments.size()) {
            simulate_plan = true;
            if (!parseSizeT(parsed.arguments[++i], simulate_limit)) {
                std::cerr << "Error: invalid value for --simulate-limit\n";
                return 1;
            }
        } else if (arg.rfind("--simulate-limit=", 0) == 0) {
            simulate_plan = true;
            if (!parseSizeT(arg.substr(17), simulate_limit)) {
                std::cerr << "Error: invalid value for --simulate-limit\n";
                return 1;
            }
        } else if (arg == "--execute") {
            execute_plan = true;
        } else if (arg == "--position" && i + 1 < parsed.arguments.size()) {
            if (!parseSizeT(parsed.arguments[++i], sequence_position)) {
                std::cerr << "Error: invalid value for --position\n";
                return 1;
            }
        } else if (arg.rfind("--position=", 0) == 0) {
            if (!parseSizeT(arg.substr(11), sequence_position)) {
                std::cerr << "Error: invalid value for --position\n";
                return 1;
            }
        } else {
            positional.push_back(arg);
        }
    }

    if (positional.empty()) {
        std::cerr << "Error: run command requires a GGUF file path\n";
        std::cerr << "Usage: mlc run [options] <gguf_path> [token_id]\n";
        return 1;
    }

    const std::string& gguf_path = positional[0];
    if (positional.size() >= 2) {
        if (!parseUnsigned(positional[1], token_id)) {
            std::cerr << "Error: token id must be an integer\n";
            return 1;
        }
    }

    if (preview_len == 0) {
        std::cerr << "Error: preview length must be greater than zero\n";
        return 1;
    }

    try {
        mlc::runtime::ModelRunner runner(gguf_path);
        mlc::runtime::RunConfig config;
        config.token_id = token_id;
        config.preview_length = preview_len;
        config.try_logits = try_logits;
        config.simulate_plan = simulate_plan;
        config.simulate_limit = simulate_limit;
        config.execute_plan = execute_plan;
        config.sequence_position = sequence_position;
        auto report = runner.dryRun(config);

        std::cout << "Model: " << gguf_path << "\n";
        std::cout << "Layers: " << report.num_layers
                  << "  Hidden: " << report.hidden_size << "\n";
        std::cout << "Execution nodes: " << report.plan.nodes().size() << "\n";
        if (!report.schedule.empty()) {
            std::cout << "Schedule sample: ";
            size_t sample = std::min<size_t>(6, report.schedule.size());
            for (size_t i = 0; i < sample; ++i) {
                if (i > 0) std::cout << " -> ";
                std::cout << report.schedule[i];
            }
            if (sample < report.schedule.size()) {
                std::cout << " -> ...";
            }
            std::cout << "\n";
        }
        if (parsed.verbose) {
            std::cout << report.plan.dump() << "\n";
        }

        std::cout << "Embedding tensor: " << report.embedding_tensor
                  << " (token " << report.token_id
                  << ", dim " << report.embedding_dim << ")\n";
        std::cout << "  values: ";
        printFloatPreview(report.embedding_preview);

        if (config.try_logits) {
            if (report.logits_tensor.empty()) {
                std::cout << "No F32 output tensor found; skipping logits pass.\n";
            } else if (!report.logits_error.empty()) {
                std::cout << "Failed to run '" << report.logits_tensor
                          << "': " << report.logits_error << "\n";
            } else {
                std::cout << "Logits tensor: " << report.logits_tensor
                          << " (rows " << report.logits_dim << ")\n";
                std::cout << "  values: ";
                printFloatPreview(report.logits_preview);
            }
        }
        if (simulate_plan && !report.execution_trace.empty()) {
            std::cout << "Execution trace:\n";
            for (const auto& line : report.execution_trace) {
                std::cout << "  - " << line << "\n";
            }
        }
        if (report.execution_ran) {
            if (!report.execution_success) {
                std::cout << "Graph execution failed: " << report.execution_error << "\n";
            } else {
                std::cout << "Graph execution output preview: ";
                printFloatPreview(report.execution_output_preview);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

int handleDecodeCommand(const mlc::util::ParsedArgs& parsed) {
    std::vector<std::string> positional;
    size_t start_pos = 0;
    size_t max_steps = 0;
    size_t top_k = 0;
    bool evict_on_full = true;
    bool cache_report = false;

    for (size_t i = 0; i < parsed.arguments.size(); ++i) {
        const std::string& arg = parsed.arguments[i];
        if (arg == "--start" && i + 1 < parsed.arguments.size()) {
            if (!parseSizeT(parsed.arguments[++i], start_pos)) {
                std::cerr << "Error: invalid value for --start\n";
                return 1;
            }
        } else if (arg == "--max-steps" && i + 1 < parsed.arguments.size()) {
            if (!parseSizeT(parsed.arguments[++i], max_steps)) {
                std::cerr << "Error: invalid value for --max-steps\n";
                return 1;
            }
        } else if (arg == "--topk" && i + 1 < parsed.arguments.size()) {
            if (!parseSizeT(parsed.arguments[++i], top_k)) {
                std::cerr << "Error: invalid value for --topk\n";
                return 1;
            }
        } else if (arg.rfind("--start=", 0) == 0) {
            if (!parseSizeT(arg.substr(8), start_pos)) {
                std::cerr << "Error: invalid value for --start\n";
                return 1;
            }
        } else if (arg.rfind("--max-steps=", 0) == 0) {
            if (!parseSizeT(arg.substr(12), max_steps)) {
                std::cerr << "Error: invalid value for --max-steps\n";
                return 1;
            }
        } else if (arg.rfind("--topk=", 0) == 0) {
            if (!parseSizeT(arg.substr(7), top_k)) {
                std::cerr << "Error: invalid value for --topk\n";
                return 1;
            }
        } else if (arg == "--no-evict") {
            evict_on_full = false;
        } else if (arg == "--cache-report") {
            cache_report = true;
        } else {
            positional.push_back(arg);
        }
    }

    if (positional.size() < 2) {
        std::cerr << "Usage: mlc decode <gguf_path> <token-ids|seq1;seq2> "
                     "[--start <pos>] [--max-steps <n>] [--topk <k>]\n";
        return 1;
    }

    const std::string& gguf_path = positional[0];
    std::vector<std::vector<uint64_t>> batches;
    try {
        batches = parseTokenBatches(positional[1]);
    } catch (const std::exception& e) {
        std::cerr << "Error parsing tokens: " << e.what() << "\n";
        return 1;
    }

    auto printCacheReport = [&](const std::vector<mlc::runtime::CacheReportEntry>& report) {
        if (report.empty()) return;
        std::cout << "KV cache report (" << report.size() << " tensors):\n";
        for (const auto& entry : report) {
            std::cout << "  - " << entry.name
                      << " dtype=" << mlc::frontend::ggufDtypeToString(entry.dtype)
                      << " qv=" << entry.quant_version
                      << " row_stride=" << entry.row_stride_bytes
                      << " bytes=" << entry.byte_size
                      << "\n";
        }
    };

    try {
        mlc::runtime::DecodeRunner runner(gguf_path);
        size_t preview = 8;

        if (batches.size() == 1) {
            mlc::runtime::DecodeOptions opts;
            opts.tokens = std::move(batches.front());
            opts.start_position = start_pos;
            opts.max_steps = max_steps;
            opts.top_k = top_k;
            opts.evict_on_full = evict_on_full;
            opts.cache_report = cache_report;
            auto res = runner.run(opts);
            printDecodeResult(res, preview);
            if (cache_report) {
                printCacheReport(res.cache_report);
            }
            return res.success ? 0 : 1;
        }

        mlc::runtime::DecodeBatchOptions opts;
        opts.sequences = std::move(batches);
        opts.start_position = start_pos;
        opts.max_steps = max_steps;
        opts.top_k = top_k;
        opts.evict_on_full = evict_on_full;
        opts.cache_report = cache_report;
        auto batch_res = runner.runBatch(opts);

        if (!batch_res.success) {
            std::cout << "Batch decode failed after processing "
                      << batch_res.results.size() << " sequences.\n";
            return 1;
        }

        for (size_t i = 0; i < batch_res.results.size(); ++i) {
            std::cout << "Sequence " << i << ":\n";
            printDecodeResult(batch_res.results[i], preview);
            if (cache_report) {
                printCacheReport(batch_res.results[i].cache_report);
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

// Forward declare llama.cpp-backed chat for fallback routing.
int handleChatLlamaCommand(const std::vector<std::string>& args);

int handleChatCommand(const std::vector<std::string>& args) {
    if (args.empty()) {
        std::cerr << "Usage: mlc chat <gguf_path> --prompt-tokens <id,id,...> [--max-new N] "
                     "[--temperature T] [--topk K] [--topp P] [--start POS] [--no-evict] [--eos ids]\n"
                     "Optional: --prompt-text \"...\" [--tokenizer-cmd '<cmd with {text} placeholder>']\n"
                     "Optional: --detokenizer-cmd '<cmd with {ids} placeholder>' --decode-output\n";
        return 1;
    }

    const std::string& gguf_path = args[0];
    std::vector<uint64_t> prompt_tokens;
    std::string prompt_text;
    std::vector<uint64_t> eos_tokens;
    size_t max_new = 64;
    float temperature = 1.0f;
    size_t top_k = 40;
    float top_p = 0.9f;
    size_t start_pos = 0;
    bool evict_on_full = true;
    bool decode_output = false;
    std::string tokenizer_cmd;
    std::string detokenizer_cmd;
    bool raw_prompt = false;
    bool use_llama_backend = false;

    for (size_t i = 1; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if ((arg == "--prompt-tokens" || arg == "--prompt") && i + 1 < args.size()) {
            prompt_tokens = parseTokenList(args[++i]);
        } else if (arg.rfind("--prompt-tokens=", 0) == 0) {
            prompt_tokens = parseTokenList(arg.substr(16));
        } else if ((arg == "--prompt-text" || arg == "--text") && i + 1 < args.size()) {
            prompt_text = args[++i];
            decode_output = true;
        } else if (arg.rfind("--prompt-text=", 0) == 0) {
            prompt_text = arg.substr(14);
            decode_output = true;
        } else if (arg == "--max-new" && i + 1 < args.size()) {
            if (!parseSizeT(args[++i], max_new)) {
                std::cerr << "Error: invalid value for --max-new\n";
                return 1;
            }
        } else if (arg.rfind("--max-new=", 0) == 0) {
            if (!parseSizeT(arg.substr(10), max_new)) {
                std::cerr << "Error: invalid value for --max-new\n";
                return 1;
            }
        } else if (arg == "--temperature" && i + 1 < args.size()) {
            try {
                temperature = std::stof(args[++i]);
            } catch (...) {
                std::cerr << "Error: invalid value for --temperature\n";
                return 1;
            }
        } else if (arg.rfind("--temperature=", 0) == 0) {
            try {
                temperature = std::stof(arg.substr(14));
            } catch (...) {
                std::cerr << "Error: invalid value for --temperature\n";
                return 1;
            }
        } else if (arg == "--topk" && i + 1 < args.size()) {
            if (!parseSizeT(args[++i], top_k)) {
                std::cerr << "Error: invalid value for --topk\n";
                return 1;
            }
        } else if (arg.rfind("--topk=", 0) == 0) {
            if (!parseSizeT(arg.substr(7), top_k)) {
                std::cerr << "Error: invalid value for --topk\n";
                return 1;
            }
        } else if (arg == "--topp" && i + 1 < args.size()) {
            try {
                top_p = std::stof(args[++i]);
            } catch (...) {
                std::cerr << "Error: invalid value for --topp\n";
                return 1;
            }
        } else if (arg.rfind("--topp=", 0) == 0) {
            try {
                top_p = std::stof(arg.substr(7));
            } catch (...) {
                std::cerr << "Error: invalid value for --topp\n";
                return 1;
            }
        } else if (arg == "--start" && i + 1 < args.size()) {
            if (!parseSizeT(args[++i], start_pos)) {
                std::cerr << "Error: invalid value for --start\n";
                return 1;
            }
        } else if (arg.rfind("--start=", 0) == 0) {
            if (!parseSizeT(arg.substr(8), start_pos)) {
                std::cerr << "Error: invalid value for --start\n";
                return 1;
            }
        } else if (arg == "--no-evict") {
            evict_on_full = false;
        } else if (arg == "--eos" && i + 1 < args.size()) {
            eos_tokens = parseTokenList(args[++i]);
        } else if (arg.rfind("--eos=", 0) == 0) {
            eos_tokens = parseTokenList(arg.substr(6));
        } else if (arg == "--decode-output") {
            decode_output = true;
        } else if (arg == "--no-decode") {
            decode_output = false;
        } else if (arg == "--tokenizer-cmd" && i + 1 < args.size()) {
            tokenizer_cmd = args[++i];
        } else if (arg.rfind("--tokenizer-cmd=", 0) == 0) {
            tokenizer_cmd = arg.substr(16);
        } else if (arg == "--detokenizer-cmd" && i + 1 < args.size()) {
            detokenizer_cmd = args[++i];
        } else if (arg.rfind("--detokenizer-cmd=", 0) == 0) {
            detokenizer_cmd = arg.substr(19);
        } else if (arg == "--raw-prompt") {
            raw_prompt = true;
        } else if (arg == "--llama-backend") {
            use_llama_backend = true;
        } else {
            std::cerr << "Unknown chat option: " << arg << "\n";
            return 1;
        }
    }

    // Fast-path fallback to llama.cpp only when explicitly requested.
    if (prompt_tokens.empty() && use_llama_backend) {
#ifdef MLC_ENABLE_LLAMA_TOKENIZER
        std::string text_for_tok = prompt_text;
        if (!raw_prompt) {
            if (text_for_tok.empty()) text_for_tok = "Hello";
            text_for_tok = "[INST] <<SYS>>You are a helpful assistant.<</SYS>>\n" + text_for_tok + " [/INST]";
        }
        std::vector<std::string> llama_args;
        llama_args.push_back(gguf_path);
        llama_args.push_back("--prompt");
        llama_args.push_back(text_for_tok.empty() ? prompt_text : text_for_tok);
        llama_args.push_back("--max-new");
        llama_args.push_back(std::to_string(max_new));
        llama_args.push_back("--temperature");
        llama_args.push_back(std::to_string(temperature));
        llama_args.push_back("--topk");
        llama_args.push_back(std::to_string(top_k));
        llama_args.push_back("--topp");
        llama_args.push_back(std::to_string(top_p));
        if (raw_prompt) llama_args.push_back("--raw");
        return handleChatLlamaCommand(llama_args);
#else
        std::cerr << "Llama backend not available; build with llama.cpp to enable --llama-backend\n";
        return 1;
#endif
    }

    try {
        mlc::runtime::Session session(gguf_path);
        mlc::runtime::Tokenizer tokenizer(session.loader());

        std::string text_for_tok = prompt_text;
        if (!raw_prompt) {
            if (text_for_tok.empty()) {
                text_for_tok = "Hello";
            }
            text_for_tok = "[INST] <<SYS>>You are a helpful assistant.<</SYS>>\n" + text_for_tok + " [/INST]";
        }

        // Prefer external tokenizer when prompt text is provided and command is set.
        if (prompt_tokens.empty() && !prompt_text.empty()) {
            if (!tokenizer_cmd.empty()) {
                prompt_tokens = tokenizeExternal(tokenizer_cmd, prompt_text);
                if (prompt_tokens.empty()) {
                    std::cerr << "External tokenizer returned no tokens; falling back to internal\n";
                }
            }
            if (prompt_tokens.empty()) {
                if (!tokenizer.valid()) {
                    std::cerr << "Error: tokenizer not available; provide --prompt-tokens or --tokenizer-cmd\n";
                    return 1;
                }
                mlc::runtime::TokenizerConfig tcfg;
                tcfg.add_bos = true;
                tcfg.add_eos = false;
                prompt_tokens = tokenizer.encode(text_for_tok, tcfg);
            }
        }

        if (prompt_tokens.empty()) {
            std::cerr << "Error: provide --prompt-text or --prompt-tokens\n";
            return 1;
        }
        const auto& log_cfg = logitsTraceConfig();
        logTokenList(log_cfg, "prompt_tokens", 0, prompt_tokens);
        auto graph = mlc::runtime::ExecutionPlanBuilder::BuildFromLoader(session.loader());
        const size_t context_len = std::max<size_t>(1, graph.modelConfig().context_length);
        mlc::runtime::ExecutionContext context(session, &graph);
        mlc::runtime::ExecutionExecutor executor(graph, &mlc::runtime::BackendRegistry::Default(), &context);

        auto setTokenInputs = [&](uint64_t token, size_t pos) {
            context.setToken(token);
            context.setSequencePosition(pos);
            static const std::vector<std::string> kTokenInputs = {"tokens", "token_ids"};
            for (const auto& name : kTokenInputs) {
                if (graph.tensors().count(name)) {
                    context.setTensor(name, {static_cast<float>(token)});
                }
            }
        };

        auto runStep = [&](uint64_t token, size_t pos, std::vector<float>& logits_out) -> bool {
            setTokenInputs(token, pos);
            auto res = executor.run();
            if (!res.success) {
                if (!res.trace.empty()) {
                    std::cerr << "Execution failed: " << formatTraceEntry(res.trace.back()) << "\n";
                } else {
                    std::cerr << "Execution failed\n";
                }
                return false;
            }
            const auto* logits = context.getTensor("logits");
            if (!logits || logits->empty()) {
                std::cerr << "Logits not produced for token " << token << "\n";
                return false;
            }
            logits_out = *logits;
            return true;
        };

        // Prime with prompt tokens.
        size_t pos = start_pos;
        std::vector<float> logits;
        for (size_t i = 0; i < prompt_tokens.size(); ++i) {
            if (pos >= context_len) {
                if (!evict_on_full) {
                    std::cerr << "Context length exceeded during prompt; use --no-evict only for short prompts\n";
                    return 1;
                }
                context.clearStateTensors();
                pos = 0;
            }
            if (!runStep(prompt_tokens[i], pos, logits)) {
                return 1;
            }
            logLogits(log_cfg, "prompt", 0, i, prompt_tokens[i], pos, logits, top_k);
            pos += 1;
        }

        mlc::runtime::SamplerOptions sampler_opts;
        sampler_opts.temperature = temperature;
        sampler_opts.top_k = top_k;
        sampler_opts.top_p = top_p;
        std::mt19937 rng(std::random_device{}());

        std::vector<uint8_t> control_mask;
        if (tokenizer.valid()) {
            size_t vocab = tokenizer.vocabSize();
            control_mask.assign(vocab, 0);
            for (size_t id = 0; id < vocab; ++id) {
                if (tokenizer.isControlToken(id) && !tokenizer.isEogToken(id)) {
                    control_mask[id] = 1;
                }
            }
        }

        std::vector<uint64_t> generated;
        for (size_t i = 0; i < max_new; ++i) {
            const std::vector<float>* sample_logits = &logits;
            std::vector<float> masked;
            if (!control_mask.empty() && logits.size() == control_mask.size()) {
                masked = logits;
                for (size_t id = 0; id < masked.size(); ++id) {
                    if (control_mask[id]) {
                        masked[id] = -1e9f;
                    }
                }
                sample_logits = &masked;
            }
            uint64_t next = mlc::runtime::sampleLogits(*sample_logits, sampler_opts, rng);
            if (next == std::numeric_limits<uint64_t>::max()) {
                std::cerr << "Sampling failed at step " << i << "\n";
                break;
            }
            generated.push_back(next);
            std::cout << next;
            if (i + 1 < max_new) std::cout << ", ";
            bool hit_eos = std::find(eos_tokens.begin(), eos_tokens.end(), next) != eos_tokens.end();
            if (hit_eos) {
                std::cout << " [eos]\n";
                break;
            }

            if (pos >= context_len) {
                if (evict_on_full) {
                    context.clearStateTensors();
                    pos = 0;
                } else {
                    std::cout << "\nReached context limit; stopping\n";
                    break;
                }
            }
            if (!runStep(next, pos, logits)) {
                break;
            }
            logLogits(log_cfg, "gen", 0, i, next, pos, logits, top_k);
            pos += 1;
        }
        if (!generated.empty() && decode_output) {
            std::cout << "\n";
            std::string decoded;
            if (!detokenizer_cmd.empty()) {
                decoded = detokenizeExternal(detokenizer_cmd, generated);
            } else if (tokenizer.valid()) {
                decoded = tokenizer.decode(generated);
            }
            if (!decoded.empty()) {
                std::cout << "Decoded: " << decoded << "\n";
            } else {
                std::cout << "Decoded: [unavailable]\n";
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

int handleChatReplCommand(const std::vector<std::string>& args) {
    if (args.empty()) {
        std::cerr << "Usage: mlc chat-repl <gguf_path> [--max-new N] [--temperature T] [--topk K] [--topp P]\n"
                     "                        [--system \"...\"] [--no-system] [--no-template] [--print-ids]\n";
        return 1;
    }

    const std::string& gguf_path = args[0];
    size_t max_new = 64;
    float temperature = 0.8f;
    size_t top_k = 40;
    float top_p = 0.9f;
    size_t start_pos = 0;
    bool print_ids = false;
    bool use_template = true;
    bool include_system = true;
    std::string system_message = "You are a helpful assistant.";
    for (size_t i = 1; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (arg == "--max-new" && i + 1 < args.size()) {
            parseSizeT(args[++i], max_new);
        } else if (arg.rfind("--max-new=", 0) == 0) {
            parseSizeT(arg.substr(10), max_new);
        } else if (arg == "--temperature" && i + 1 < args.size()) {
            temperature = std::stof(args[++i]);
        } else if (arg.rfind("--temperature=", 0) == 0) {
            temperature = std::stof(arg.substr(14));
        } else if (arg == "--topk" && i + 1 < args.size()) {
            parseSizeT(args[++i], top_k);
        } else if (arg.rfind("--topk=", 0) == 0) {
            parseSizeT(arg.substr(7), top_k);
        } else if (arg == "--topp" && i + 1 < args.size()) {
            top_p = std::stof(args[++i]);
        } else if (arg.rfind("--topp=", 0) == 0) {
            top_p = std::stof(arg.substr(7));
        } else if (arg == "--print-ids") {
            print_ids = true;
        } else if ((arg == "--system" || arg == "--system-message") && i + 1 < args.size()) {
            system_message = args[++i];
        } else if (arg.rfind("--system=", 0) == 0) {
            system_message = arg.substr(9);
        } else if (arg == "--no-system") {
            include_system = false;
        } else if (arg == "--no-template" || arg == "--raw") {
            use_template = false;
        }
    }

    try {
        mlc::runtime::Session session(gguf_path);
        mlc::runtime::Tokenizer tokenizer(session.loader());
        if (!tokenizer.valid()) {
            std::cerr << "Tokenizer not available in GGUF\n";
            return 1;
        }

        std::string chat_template;
        if (use_template) {
            const auto& kv = session.loader().kvMetadata();
            auto it = kv.find("tokenizer.chat_template");
            if (it != kv.end() && it->second.type == mlc::frontend::GGUFValueType::STRING) {
                chat_template = std::get<std::string>(it->second.data);
            }
        }
        const bool can_template = use_template && !chat_template.empty();
        bool warned_template = false;

        auto graph = mlc::runtime::ExecutionPlanBuilder::BuildFromLoader(session.loader());
        const size_t context_len = std::max<size_t>(1, graph.modelConfig().context_length);
        mlc::runtime::ExecutionContext context(session, &graph);
        mlc::runtime::ExecutionExecutor executor(graph, &mlc::runtime::BackendRegistry::Default(), &context);

        auto setTokenInputs = [&](uint64_t token, size_t pos) {
            context.setToken(token);
            context.setSequencePosition(pos);
            static const std::vector<std::string> kTokenInputs = {"tokens", "token_ids"};
            for (const auto& name : kTokenInputs) {
                if (graph.tensors().count(name)) {
                    context.setTensor(name, {static_cast<float>(token)});
                }
            }
        };

        auto runStep = [&](uint64_t token, size_t pos, std::vector<float>& logits_out) -> bool {
            setTokenInputs(token, pos);
            auto res = executor.run();
            if (!res.success) {
                std::cerr << "Execution failed\n";
                return false;
            }
            const auto* logits = context.getTensor("logits");
            if (!logits || logits->empty()) {
                std::cerr << "Logits not produced\n";
                return false;
            }
            logits_out = *logits;
            return true;
        };

        mlc::runtime::SamplerOptions sampler_opts;
        sampler_opts.temperature = temperature;
        sampler_opts.top_k = top_k;
        sampler_opts.top_p = top_p;
        std::mt19937 rng(std::random_device{}());
        const auto& chat_trace = chatTraceConfig();
        const auto& log_cfg = logitsTraceConfig();

        std::vector<uint8_t> control_mask;
        if (tokenizer.valid()) {
            size_t vocab = tokenizer.vocabSize();
            control_mask.assign(vocab, 0);
            for (size_t id = 0; id < vocab; ++id) {
                if (tokenizer.isControlToken(id) && !tokenizer.isEogToken(id)) {
                    control_mask[id] = 1;
                }
            }
        }

        std::vector<uint64_t> history;
        size_t processed = 0;
        if (!can_template) {
            auto bos = tokenizer.bosId();
            if (bos != std::numeric_limits<uint64_t>::max()) {
                history.push_back(bos);
            }
        }

        size_t pos = start_pos;
        std::cout << "Interactive chat. Type 'exit' to quit.\n";
        std::vector<ChatMessage> messages;
        if (include_system && !system_message.empty()) {
            messages.push_back({"system", system_message});
        }
        std::string line;
        bool first_turn = true;
        size_t turn_index = 0;
        while (true) {
            std::cout << "You: ";
            if (!std::getline(std::cin, line)) break;
            if (line == "exit" || line == "quit") break;

            messages.push_back({"user", line});
            bool used_template = false;
#ifdef MLC_ENABLE_LLAMA_TOKENIZER
#endif
            if (can_template && !used_template) {
                std::string rendered = renderChatTemplateSimple(chat_template, messages, true, tokenizer);
                if (!rendered.empty()) {
                    mlc::runtime::TokenizerConfig cfg;
                    cfg.add_bos = !templateHasBosHint(chat_template);
                    cfg.add_eos = false;
                    history = tokenizer.encode(rendered, cfg);
                    processed = 0;
                    pos = start_pos;
                    context.clearStateTensors();
                    used_template = true;
                    logChatPrompt(chat_trace, turn_index, true, rendered);
                    logTokenListToFile(chat_trace.tokens_file, "chat_prompt_tokens", turn_index, history);
                }
            }
#ifdef MLC_ENABLE_LLAMA_TOKENIZER
            if (can_template && !used_template) {
                std::string err;
                std::string rendered = applyChatTemplateLlama(chat_template, messages, true, &err);
                if (!rendered.empty()) {
                    mlc::runtime::TokenizerConfig cfg;
                    cfg.add_bos = !templateHasBosHint(chat_template);
                    cfg.add_eos = false;
                    history = tokenizer.encode(rendered, cfg);
                    processed = 0;
                    pos = start_pos;
                    context.clearStateTensors();
                    used_template = true;
                    logChatPrompt(chat_trace, turn_index, true, rendered);
                    logTokenListToFile(chat_trace.tokens_file, "chat_prompt_tokens", turn_index, history);
                } else if (!warned_template) {
                    std::cerr << "Warning: failed to apply chat template (" << err
                              << "); falling back to legacy prompt format.\n";
                    warned_template = true;
                }
            }
#endif
            if (!used_template) {
                std::string chat_text;
                if (first_turn) {
                    if (include_system && !system_message.empty()) {
                        chat_text = "[INST] <<SYS>>" + system_message + "<</SYS>>\n" + line + " [/INST]";
                    } else {
                        chat_text = "[INST] " + line + " [/INST]";
                    }
                } else {
                    chat_text = "[INST] " + line + " [/INST]";
                }
                first_turn = false;

                mlc::runtime::TokenizerConfig cfg;
                cfg.add_bos = false;
                cfg.add_eos = false;
                auto encoded = tokenizer.encode(chat_text + "\n", cfg);
                history.insert(history.end(), encoded.begin(), encoded.end());
                logChatPrompt(chat_trace, turn_index, false, chat_text + "\n");
                logTokenListToFile(chat_trace.tokens_file, "chat_prompt_tokens", turn_index, history);
            }
            logTokenList(log_cfg, "chat_prompt_tokens", turn_index, history);

            std::vector<float> logits;
            // Feed any unprocessed tokens (including BOS once).
            for (size_t i = processed; i < history.size(); ++i) {
                if (pos >= context_len) {
                    context.clearStateTensors();
                    // Rebuild KV from scratch when reset.
                    pos = 0;
                    processed = 0;
                    i = processed;
                }
                if (!runStep(history[i], pos, logits)) {
                    return 1;
                }
                logLogits(log_cfg, "prompt", turn_index, i, history[i], pos, logits, top_k);
                pos += 1;
            }
            processed = history.size();

            std::vector<uint64_t> generated;
            for (size_t i = 0; i < max_new; ++i) {
                const std::vector<float>* sample_logits = &logits;
                std::vector<float> masked;
                if (!control_mask.empty() && logits.size() == control_mask.size()) {
                    masked = logits;
                    for (size_t id = 0; id < masked.size(); ++id) {
                        if (control_mask[id]) {
                            masked[id] = -1e9f;
                        }
                    }
                    sample_logits = &masked;
                }
                uint64_t next = mlc::runtime::sampleLogits(*sample_logits, sampler_opts, rng);
                if (next == std::numeric_limits<uint64_t>::max()) break;
                if (next == tokenizer.eosId()) {
                    break;
                }
                generated.push_back(next);
                history.push_back(next);
                if (pos >= context_len) {
                    context.clearStateTensors();
                    pos = 0;
                    processed = 0;
                    // Rebuild KV and logits.
                    std::vector<float> tmp_logits;
                    for (size_t j = 0; j < history.size(); ++j) {
                        if (!runStep(history[j], pos, tmp_logits)) break;
                        pos += 1;
                    }
                    logits = tmp_logits;
                    processed = history.size();
                    continue;
                }
                if (!runStep(next, pos, logits)) break;
                logLogits(log_cfg, "gen", turn_index, i, next, pos, logits, top_k);
                pos += 1;
            }
            if (print_ids) {
                std::cout << "Model ids: ";
                for (size_t i = 0; i < generated.size(); ++i) {
                    if (i) std::cout << ", ";
                    std::cout << generated[i];
                }
                std::cout << "\n";
            }
            std::string decoded = tokenizer.decode(generated);
            std::cout << "Model: " << (decoded.empty() ? "[empty]" : decoded) << "\n";
            if (used_template && !decoded.empty()) {
                messages.push_back({"assistant", decoded});
            }
            processed = history.size();
            turn_index += 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

int handleChatLlamaCommand(const std::vector<std::string>& args) {
    if (args.empty()) {
        std::cerr << "Usage: mlc chat-llama <gguf_path> --prompt \"...\" [--max-new N] [--temperature T] [--topk K] [--topp P]\n";
        return 1;
    }
    const std::string& gguf_path = args[0];
    std::string prompt;
    size_t max_new = 128;
    float temperature = 0.7f;
    size_t top_k = 40;
    float top_p = 0.9f;
    bool raw_prompt = false;
    for (size_t i = 1; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if ((arg == "--prompt" || arg == "-p") && i + 1 < args.size()) {
            prompt = args[++i];
        } else if (arg.rfind("--prompt=", 0) == 0) {
            prompt = arg.substr(9);
        } else if (arg == "--max-new" && i + 1 < args.size()) {
            parseSizeT(args[++i], max_new);
        } else if (arg.rfind("--max-new=", 0) == 0) {
            parseSizeT(arg.substr(10), max_new);
        } else if (arg == "--temperature" && i + 1 < args.size()) {
            temperature = std::stof(args[++i]);
        } else if (arg.rfind("--temperature=", 0) == 0) {
            temperature = std::stof(arg.substr(14));
        } else if (arg == "--topk" && i + 1 < args.size()) {
            parseSizeT(args[++i], top_k);
        } else if (arg.rfind("--topk=", 0) == 0) {
            parseSizeT(arg.substr(7), top_k);
        } else if (arg == "--topp" && i + 1 < args.size()) {
            top_p = std::stof(args[++i]);
        } else if (arg.rfind("--topp=", 0) == 0) {
            top_p = std::stof(arg.substr(7));
        } else if (arg == "--raw") {
            raw_prompt = true;
        }
    }
    if (!raw_prompt) {
        if (prompt.empty()) prompt = "Hello";
        prompt = "[INST] <<SYS>>You are a helpful assistant.<</SYS>>\n" + prompt + " [/INST]";
    }

#ifdef MLC_ENABLE_LLAMA_TOKENIZER
    llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap = true;
    llama_model* model = llama_model_load_from_file(gguf_path.c_str(), mparams);
    if (!model) {
        std::cerr << "Failed to load model via llama.cpp\n";
        return 1;
    }
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    llama_context* ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) {
        std::cerr << "Failed to create context\n";
        llama_model_free(model);
        return 1;
    }

    auto vocab = llama_model_get_vocab(model);
    // Tokenize prompt
    int32_t needed = llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                                    nullptr, 0, true, false);
    if (needed < 0) needed = -needed;
    std::vector<llama_token> tokens(static_cast<size_t>(needed));
    int32_t written = llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                                     tokens.data(), needed, true, false);
    if (written < 0) written = -written;
    tokens.resize(static_cast<size_t>(written));

    const auto& log_cfg = logitsTraceConfig();
    const auto& emb_cfg = embeddingsTraceConfig();
    if (emb_cfg.file) {
        llama_set_embeddings(ctx, true);
    }
    size_t pos = 0;
    std::vector<uint64_t> prompt_tokens;
    prompt_tokens.reserve(tokens.size());
    for (llama_token t : tokens) {
        prompt_tokens.push_back(static_cast<uint64_t>(t));
    }
    if (log_cfg.file) {
        logTokenList(log_cfg, "prompt_tokens", 0, prompt_tokens);
    }
    if (emb_cfg.file) {
        logTokenListToFile(emb_cfg.file, "prompt_tokens", 0, prompt_tokens);
    }
    if (log_cfg.file || emb_cfg.file) {
        int32_t n_embd = llama_model_n_embd(model);
        for (size_t i = 0; i < tokens.size(); ++i) {
            llama_token tok = tokens[i];
            llama_batch prompt_batch = llama_batch_get_one(&tok, 1);
            if (llama_decode(ctx, prompt_batch) != 0) {
                std::cerr << "llama_decode failed on prompt\n";
                llama_free(ctx);
                llama_model_free(model);
                return 1;
            }
            const float* logits_ptr = llama_get_logits(ctx);
            int64_t n_vocab = llama_vocab_n_tokens(vocab);
            std::vector<float> logits(logits_ptr, logits_ptr + n_vocab);
            if (log_cfg.file) {
                logLogits(log_cfg, "prompt", 0, i, static_cast<uint64_t>(tok), pos, logits, top_k);
            }
            if (emb_cfg.file) {
                float* emb = llama_get_embeddings_ith(ctx, -1);
                if (emb) {
                    logEmbeddings(emb_cfg, "prompt", 0, i, static_cast<uint64_t>(tok), pos, emb, n_embd);
                }
            }
            pos += 1;
        }
    } else {
        // Evaluate prompt (auto-tracked positions)
        llama_batch prompt_batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
        if (llama_decode(ctx, prompt_batch) != 0) {
            std::cerr << "llama_decode failed on prompt\n";
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        pos = tokens.size();
    }

    auto sample_next = [&](const std::vector<float>& logits_vec) -> llama_token {
        // simple top-k/top-p sampling with temperature
        std::vector<std::pair<float, llama_token>> scored;
        scored.reserve(logits_vec.size());
        for (int i = 0; i < (int)logits_vec.size(); ++i) {
            scored.emplace_back(logits_vec[i] / std::max(temperature, 1e-5f), (llama_token)i);
        }
        std::partial_sort(scored.begin(),
                          scored.begin() + std::min(top_k, scored.size()),
                          scored.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });
        if (top_k < scored.size()) scored.resize(top_k);
        // top-p cutoff
        float max_logit = scored.front().first;
        float sum = 0.f;
        for (auto& kv : scored) kv.first = std::exp(kv.first - max_logit), sum += kv.first;
        std::vector<std::pair<float, llama_token>> filtered;
        float acc = 0.f;
        for (auto& kv : scored) {
            float p = kv.first / sum;
            acc += p;
            filtered.emplace_back(p, kv.second);
            if (acc >= top_p) break;
        }
        float r = (float)rand() / (float)RAND_MAX;
        float cur = 0.f;
        for (auto& kv : filtered) {
            cur += kv.first;
            if (r <= cur) return kv.second;
        }
        return filtered.back().second;
    };

    std::vector<llama_token> generated;
    for (size_t i = 0; i < max_new; ++i) {
        const float* logits_ptr = llama_get_logits(ctx);
        auto vocab = llama_model_get_vocab(model);
        int64_t n_vocab = llama_vocab_n_tokens(vocab);
        std::vector<float> logits(logits_ptr, logits_ptr + n_vocab);
        if (log_cfg.file) {
            uint64_t prev_token = prompt_tokens.empty() ? 0
                                                        : static_cast<uint64_t>(prompt_tokens.back());
            if (!generated.empty()) {
                prev_token = static_cast<uint64_t>(generated.back());
            }
            logLogits(log_cfg, "gen", 0, i, prev_token, pos, logits, top_k);
        }
        llama_token next = sample_next(logits);
        if (next == llama_vocab_eos(vocab)) break;
        generated.push_back(next);
        llama_batch gen_batch = llama_batch_get_one(&next, 1);
        if (llama_decode(ctx, gen_batch) != 0) break;
        if (emb_cfg.file) {
            int32_t n_embd = llama_model_n_embd(model);
            float* emb = llama_get_embeddings_ith(ctx, -1);
            if (emb) {
                logEmbeddings(emb_cfg, "gen", 0, i, static_cast<uint64_t>(next), pos, emb, n_embd);
            }
        }
        pos += 1;
    }

    // Detokenize
    int32_t guess = (int32_t)generated.size() * 8 + 32;
    std::string decoded;
    std::vector<char> buf((size_t)guess);
    int32_t det = llama_detokenize(vocab,
                                   generated.data(),
                                   (int32_t)generated.size(),
                                   buf.data(),
                                   (int32_t)buf.size(),
                                   true,
                                   false);
    if (det < 0) {
        int32_t need = -det;
        buf.resize((size_t)need);
        det = llama_detokenize(vocab,
                               generated.data(),
                               (int32_t)generated.size(),
                               buf.data(),
                               (int32_t)buf.size(),
                               true,
                               false);
    }
    if (det > 0) decoded.assign(buf.data(), (size_t)det);
    std::cout << decoded << "\n";

    llama_free(ctx);
    llama_model_free(model);
    return 0;
#else
    std::cerr << "chat-llama requires llama.cpp integration\n";
    return 1;
#endif
}

int handleTokenizeCommand(const std::vector<std::string>& args) {
    if (args.size() < 2) {
        std::cerr << "Usage: mlc tokenize <gguf_path> <text> [--no-bos] [--eos] "
                     "[--tokenizer-cmd '<cmd with {text} placeholder>'] "
                     "[--detokenizer-cmd '<cmd with {ids} placeholder>']\n";
        return 1;
    }
    const std::string& gguf_path = args[0];
    std::string text = args[1];
    bool add_bos = true;
    bool add_eos = false;
    std::string tokenizer_cmd;
    std::string detokenizer_cmd;
    for (size_t i = 2; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (arg == "--no-bos") {
            add_bos = false;
        } else if (arg == "--eos") {
            add_eos = true;
        } else if (arg.rfind("--text=", 0) == 0) {
            text = arg.substr(7);
        } else if (arg == "--tokenizer-cmd" && i + 1 < args.size()) {
            tokenizer_cmd = args[++i];
        } else if (arg.rfind("--tokenizer-cmd=", 0) == 0) {
            tokenizer_cmd = arg.substr(16);
        } else if (arg == "--detokenizer-cmd" && i + 1 < args.size()) {
            detokenizer_cmd = args[++i];
        } else if (arg.rfind("--detokenizer-cmd=", 0) == 0) {
            detokenizer_cmd = arg.substr(19);
        } else {
            std::cerr << "Unknown option for tokenize: " << arg << "\n";
            return 1;
        }
    }

    try {
        mlc::frontend::GGUFLoader loader(gguf_path);
        loader.load();
        std::vector<uint64_t> ids;
        if (!tokenizer_cmd.empty()) {
            ids = tokenizeExternal(tokenizer_cmd, text);
        }
        mlc::runtime::Tokenizer tokenizer(loader);
        if (ids.empty()) {
            if (!tokenizer.valid()) {
                std::cerr << "Error: tokenizer metadata missing in GGUF and external tokenizer not provided\n";
                return 1;
            }
            mlc::runtime::TokenizerConfig cfg;
            cfg.add_bos = add_bos;
            cfg.add_eos = add_eos;
            ids = tokenizer.encode(text, cfg);
        }

        std::cout << "Tokens (" << ids.size() << "): ";
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << ids[i];
        }
        std::cout << "\n";
        std::string decoded;
        if (!detokenizer_cmd.empty()) {
            decoded = detokenizeExternal(detokenizer_cmd, ids);
        } else if (tokenizer.valid()) {
            decoded = tokenizer.decode(ids);
        }
        if (!decoded.empty()) {
            std::cout << "Decoded: " << decoded << "\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

int handleDetokenizeCommand(const std::vector<std::string>& args) {
    if (args.size() < 2) {
        std::cerr << "Usage: mlc detokenize <gguf_path> <id0,id1,...>\n";
        return 1;
    }
    const std::string& gguf_path = args[0];
    std::vector<uint64_t> ids;
    try {
        ids = parseTokenList(args[1]);
    } catch (const std::exception& e) {
        std::cerr << "Error parsing ids: " << e.what() << "\n";
        return 1;
    }
    try {
        mlc::frontend::GGUFLoader loader(gguf_path);
        loader.load();
        mlc::runtime::Tokenizer tokenizer(loader);
        if (!tokenizer.valid()) {
            std::cerr << "Error: tokenizer metadata missing in GGUF\n";
            return 1;
        }
        std::string decoded = tokenizer.decode(ids);
        std::cout << decoded << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

int handleCapabilitiesCommand() {
    auto& metal = mlc::runtime::MetalExecutor::Instance();
    bool device_available = metal.isAvailable();
    bool ffn_kernel = metal.hasFeedForwardKernel();
    bool add_kernel = metal.hasAddKernel();
    bool norm_kernel = metal.hasRmsNormKernel();
    bool softmax_kernel = metal.hasSoftmaxKernel();

    std::cout << "Runtime Capabilities\n";
    std::cout << "====================\n";
    std::cout << "Metal device available: " << (device_available ? "yes" : "no") << "\n";
    auto describe = [&](const char* label, bool built) {
        std::cout << "Metal " << label << " kernel: "
                  << (built ? "built" : "not built (CPU fallback)") << "\n";
    };
    describe("feed-forward", ffn_kernel);
    describe("residual add", add_kernel);
    describe("RMS norm", norm_kernel);
    describe("softmax", softmax_kernel);
    if (!device_available) {
        std::cout << "Note: kernels fall back to Accelerate/CPU when Metal queues are unavailable.\n";
    }
    return 0;
}

int handleCompareCommand(const std::vector<std::string>& args) {
    enum class Mode { None, MetalVsCpu, VsLlamacpp };
    auto usage = []() {
        std::cerr << "Usage: mlc compare <mode> <gguf_path> --prompt \"...\" [options]\n"
                     "Modes:\n"
                     "  --metal-vs-cpu              Run the model twice in-process (CPU-pinned vs Metal)\n"
                     "                                and diff per-layer tap tensors.\n"
                     "  --vs-llamacpp               Run mlc once (CPU pinned) and compare against\n"
                     "                                pre-dumped llama.cpp tensors at boundaries.\n"
                     "Options:\n"
                     "  --prompt \"<text>\"           Prompt text (required).\n"
                     "  --tap <pattern>             Substring pattern for tensors to capture.\n"
                     "                                Repeatable. Default: embedding/per-block boundaries\n"
                     "                                + final_norm + logits.\n"
                     "  --csv-out <path>            Write per-tensor metrics to CSV at <path>.\n"
                     "  --reference-dir <dir>       (vs-llamacpp only) Directory of reference tensor\n"
                     "                                dumps. Each tensor X is read from\n"
                     "                                <dir>/<sanitized(X)>.f32.bin (raw little-endian f32,\n"
                     "                                length must match).\n"
                     "  --wrap-chat                 Wrap prompt with [INST] template before tokenizing\n"
                     "                                (default off — raw prompt for parity tests).\n"
                     "\n"
                     "Env vars:\n"
                     "  MLC_PARITY_DUMP=DIR         Dump both sides' captured tap tensors to DIR as\n"
                     "                                raw little-endian float32 blobs. Files are named\n"
                     "                                <side>_<sanitized_tensor>.f32.bin where <side> is\n"
                     "                                'cpu'/'metal' (--metal-vs-cpu) or 'mlc'/'llamacpp'\n"
                     "                                (--vs-llamacpp). Use for per-head / layout checks.\n"
                     "  MLC_HARNESS_STRICT=1        (--metal-vs-cpu only) Exit nonzero if the side_a run\n"
                     "                                had dispatch leaks (nodes actually ran on Metal\n"
                     "                                under force_cpu) — turns the warning into a CI gate.\n";
    };

    if (args.empty()) { usage(); return 1; }

    Mode mode = Mode::None;
    mlc::runtime::parity::CompareOptions opts;
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        if (a == "--metal-vs-cpu") {
            if (mode != Mode::None) { std::cerr << "compare: pick one mode\n"; return 1; }
            mode = Mode::MetalVsCpu;
        } else if (a == "--vs-llamacpp") {
            if (mode != Mode::None) { std::cerr << "compare: pick one mode\n"; return 1; }
            mode = Mode::VsLlamacpp;
        } else if (a == "--prompt" && i + 1 < args.size()) {
            opts.prompt = args[++i];
        } else if (a.rfind("--prompt=", 0) == 0) {
            opts.prompt = a.substr(9);
        } else if (a == "--tap" && i + 1 < args.size()) {
            opts.tap_patterns.push_back(args[++i]);
        } else if (a.rfind("--tap=", 0) == 0) {
            opts.tap_patterns.push_back(a.substr(6));
        } else if (a == "--csv-out" && i + 1 < args.size()) {
            opts.csv_path = args[++i];
        } else if (a.rfind("--csv-out=", 0) == 0) {
            opts.csv_path = a.substr(10);
        } else if (a == "--reference-dir" && i + 1 < args.size()) {
            opts.reference_dir = args[++i];
        } else if (a.rfind("--reference-dir=", 0) == 0) {
            opts.reference_dir = a.substr(16);
        } else if (a == "--wrap-chat") {
            opts.wrap_chat_template = true;
        } else if (!a.empty() && a[0] != '-' && opts.gguf_path.empty()) {
            opts.gguf_path = a;
        } else {
            std::cerr << "compare: unknown option '" << a << "'\n";
            usage();
            return 1;
        }
    }
    if (mode == Mode::None) { std::cerr << "compare: missing --metal-vs-cpu or --vs-llamacpp\n"; usage(); return 1; }
    if (opts.gguf_path.empty()) { std::cerr << "compare: missing <gguf_path>\n"; usage(); return 1; }
    if (opts.prompt.empty()) { std::cerr << "compare: missing --prompt\n"; usage(); return 1; }
    if (mode == Mode::VsLlamacpp && opts.reference_dir.empty()) {
        std::cerr << "compare --vs-llamacpp: missing --reference-dir\n"; return 1;
    }

    try {
        mlc::runtime::parity::CompareReport report =
            (mode == Mode::MetalVsCpu)
                ? mlc::runtime::parity::compareMetalVsCpu(opts)
                : mlc::runtime::parity::compareVsLlamaCpp(opts);

        mlc::runtime::parity::printTable(report, std::cout);
        if (!opts.csv_path.empty()) {
            if (!mlc::runtime::parity::writeCsv(report, opts.csv_path)) {
                std::cerr << "compare: failed to write CSV at " << opts.csv_path << "\n";
                return 2;
            }
            std::cerr << "compare: wrote CSV to " << opts.csv_path << "\n";
        }
        return report.success ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "compare: error: " << e.what() << "\n";
        return 1;
    }
}

// ============================================================================
// test-matmul-q4: focused micro-test for metal.matmul.quant on Q4_0 weights.
// CPU-dequant + Accelerate sgemv reference vs the Metal kernel, on real
// block-0 weights and on synthetic stress shapes.
// ============================================================================

namespace q4diag {

// HANDWRITTEN reference: dequantize + cblas_sgemv driven by caller-supplied
// (rows, cols, transpose). Mirrors operator_backend.cpp's interpretation —
// rows = shape[0], cols = shape[1]. NOT a canonical anchor: it shares any
// shape-axis assumption the caller (and the Metal kernel) makes.
std::vector<float> cpuMatmulHandwritten(const uint8_t* weights,
                             size_t rows,
                             size_t cols,
                             size_t row_stride,
                             uint32_t qversion,
                             const std::vector<float>& input,
                             bool transpose) {
    std::vector<float> dq(rows * cols);
    for (size_t r = 0; r < rows; ++r) {
        mlc::runtime::dequantizeRowQ4_0(weights + r * row_stride, cols, qversion,
                                         dq.data() + r * cols);
    }
    size_t out_size = transpose ? cols : rows;
    std::vector<float> out(out_size, 0.0f);
#if defined(__APPLE__)
    cblas_sgemv(CblasRowMajor, transpose ? CblasTrans : CblasNoTrans,
                static_cast<int>(rows), static_cast<int>(cols),
                1.0f, dq.data(), static_cast<int>(cols),
                input.data(), 1,
                0.0f, out.data(), 1);
#else
    // Portable fallback (slower).
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            float w = dq[r * cols + c];
            if (transpose) out[c] += w * input[r];
            else            out[r] += w * input[c];
        }
    }
#endif
    return out;
}

// CANONICAL reference: invokes Session::runLinear, the same function the
// in-graph CPU backend calls (operator_backend.cpp:362). Encodes whatever
// shape-axis convention runLinear uses (currently GGML: cols=shape[0],
// rows=shape[1]). This is the anchor — any divergence between handwritten
// and canonical is a convention/dispatch-class bug, not a kernel bug.
std::vector<float> cpuMatmulCanonical(const mlc::runtime::Session& session,
                                       const std::string& weight_name,
                                       const std::vector<float>& input) {
    return session.runLinear(weight_name, input);
}

// Classify a 3-way pairwise comparison. Threshold 0.999 picks up any divergence
// beyond float-rounding noise (well above the ~1e-6 we see when math agrees).
//   OK                — all three pairs agree.
//   DISPATCH MISMATCH — H↔M agree but disagree with C: handwritten and Metal
//                       share the wrong convention (Bug B signature).
//   KERNEL ERROR      — H↔C agree but disagree with M: dispatch is fine,
//                       Metal kernel is wrong (Bug A signature).
//   BOTH              — neither pair agrees: kernel and dispatch both off,
//                       or some other independent failure.
const char* classifyDivergence(float h_c, float h_m, float c_m) {
    constexpr float kThreshold = 0.999f;
    auto ok = [](float x) {
        return std::isfinite(x) && x >= kThreshold;
    };
    bool hc_ok = ok(h_c);
    bool hm_ok = ok(h_m);
    bool cm_ok = ok(c_m);
    if (hc_ok && hm_ok && cm_ok) return "OK";
    if (hm_ok && !cm_ok) return "DISPATCH MISMATCH";
    if (hc_ok && !hm_ok) return "KERNEL ERROR";
    return "BOTH";
}

struct DiagMetrics {
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    float rms = 0.0f;
    float cosine = 0.0f;
    std::vector<size_t> worst_idx;       // top-K output indices by |error|
    std::vector<float>  worst_err;
    std::vector<float>  bucket_mean_err; // 16 equal-width buckets across output dim
};

DiagMetrics metrics(const std::vector<float>& a, const std::vector<float>& b, int top_k = 8) {
    DiagMetrics m;
    if (a.size() != b.size() || a.empty()) {
        m.cosine = std::numeric_limits<float>::quiet_NaN();
        return m;
    }
    double max_abs = 0, sum_abs = 0, sum_sq = 0, dot = 0, na2 = 0, nb2 = 0;
    std::vector<std::pair<float, size_t>> errs;
    errs.reserve(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        double d  = double(a[i]) - double(b[i]);
        double ad = std::fabs(d);
        if (ad > max_abs) max_abs = ad;
        sum_abs += ad;
        sum_sq  += d * d;
        dot += double(a[i]) * double(b[i]);
        na2 += double(a[i]) * double(a[i]);
        nb2 += double(b[i]) * double(b[i]);
        errs.push_back({static_cast<float>(ad), i});
    }
    m.max_abs  = static_cast<float>(max_abs);
    m.mean_abs = static_cast<float>(sum_abs / a.size());
    m.rms      = static_cast<float>(std::sqrt(sum_sq / a.size()));
    m.cosine   = (na2 > 0 && nb2 > 0)
                     ? static_cast<float>(dot / (std::sqrt(na2) * std::sqrt(nb2)))
                     : std::numeric_limits<float>::quiet_NaN();

    int K = std::min<int>(top_k, static_cast<int>(errs.size()));
    std::partial_sort(errs.begin(), errs.begin() + K, errs.end(),
                      [](const auto& p, const auto& q) { return p.first > q.first; });
    for (int i = 0; i < K; ++i) {
        m.worst_idx.push_back(errs[i].second);
        m.worst_err.push_back(errs[i].first);
    }

    constexpr int B = 16;
    std::vector<double> sums(B, 0.0);
    std::vector<int> counts(B, 0);
    for (size_t i = 0; i < a.size(); ++i) {
        int b_idx = std::min(B - 1, static_cast<int>(i * B / a.size()));
        sums[b_idx] += std::fabs(double(a[i]) - double(b[i]));
        counts[b_idx]++;
    }
    m.bucket_mean_err.resize(B);
    for (int i = 0; i < B; ++i) {
        m.bucket_mean_err[i] = counts[i] > 0
                                   ? static_cast<float>(sums[i] / counts[i])
                                   : 0.0f;
    }
    return m;
}

// Capture blk.0.attn_norm.out from a single CPU-pinned prefill of `prompt`.
std::vector<float> captureRealInput(mlc::runtime::Session& session,
                                    const std::string& prompt) {
    using namespace mlc::runtime;

    Tokenizer tokenizer(session.loader());
    if (!tokenizer.valid()) throw std::runtime_error("tokenizer init failed");
    TokenizerConfig cfg; cfg.add_bos = true; cfg.add_eos = false;
    auto tokens = tokenizer.encode(prompt, cfg);
    if (tokens.empty()) throw std::runtime_error("tokenizer produced no tokens");

    bool prior = KernelDescriptorRegistry::forceCpu();
    KernelDescriptorRegistry::setForceCpu(true);

    auto graph = ExecutionPlanBuilder::BuildFromLoader(session.loader());
    ExecutionContext context(session, &graph);
    ExecutionExecutor executor(graph, &BackendRegistry::Default(), &context);
    context.registerTap("blk.0.attn_norm.out");

    size_t pos = 0;
    for (uint64_t tok : tokens) {
        context.setToken(tok);
        context.setSequencePosition(pos);
        for (const auto& tname : {std::string("tokens"), std::string("token_ids")}) {
            if (graph.tensors().count(tname)) {
                context.setTensor(tname, {static_cast<float>(tok)});
            }
        }
        auto res = executor.run();
        if (!res.success) {
            KernelDescriptorRegistry::setForceCpu(prior);
            throw std::runtime_error("CPU prefill failed at token index " + std::to_string(pos));
        }
        ++pos;
    }
    KernelDescriptorRegistry::setForceCpu(prior);

    auto it = context.tapData().find("blk.0.attn_norm.out");
    if (it == context.tapData().end()) {
        throw std::runtime_error("blk.0.attn_norm.out was not produced");
    }
    return it->second;
}

} // namespace q4diag

int handleTestMatmulQ4Command(const std::vector<std::string>& args) {
    using namespace mlc::runtime;
    using namespace mlc::frontend;

    auto usage = []() {
        std::cerr << "Usage: mlc test-matmul-q4 <gguf_path>"
                     " [--stress] [--seed N] [--prompt \"...\"] [--synthetic-input]\n"
                     "Compares CPU-dequant+sgemv reference vs metal.matmul.quant on Q4_0 weights.\n"
                     "By default uses captured blk.0.attn_norm.out as the input vector;\n"
                     "--synthetic-input uses a deterministic random vector instead.\n"
                     "--stress varies output dim N (K=2048 fixed) for both kernel variants.\n";
    };
    if (args.empty()) { usage(); return 1; }

    std::string gguf_path;
    bool do_stress = false;
    bool synthetic_input = false;
    uint64_t seed = 42;
    std::string prompt = "The capital of France is";
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        if (a == "--stress") do_stress = true;
        else if (a == "--synthetic-input") synthetic_input = true;
        else if (a == "--seed" && i + 1 < args.size()) {
            try { seed = std::stoull(args[++i]); }
            catch (...) { std::cerr << "bad --seed\n"; return 1; }
        }
        else if (a == "--prompt" && i + 1 < args.size()) prompt = args[++i];
        else if (gguf_path.empty() && !a.empty() && a[0] != '-') gguf_path = a;
        else { std::cerr << "unknown arg: " << a << "\n"; usage(); return 1; }
    }
    if (gguf_path.empty()) { usage(); return 1; }

    try {
        Session session(gguf_path);
        const auto& tensors = session.loader().tensors();
        uint32_t qversion = session.loader().quantizationVersion();

        // Input vector (size K=2048 — TinyLlama hidden_dim).
        std::vector<float> input;
        if (synthetic_input) {
            input.resize(2048);
            std::mt19937_64 rng(seed);
            std::normal_distribution<float> dist(0.0f, 0.1f);
            for (auto& v : input) v = dist(rng);
            std::printf("# input source: synthetic N(0,0.1) seed=%llu\n",
                        static_cast<unsigned long long>(seed));
        } else {
            std::printf("# input source: captured blk.0.attn_norm.out from CPU prefill of \"%s\"\n",
                        prompt.c_str());
            input = q4diag::captureRealInput(session, prompt);
        }
        if (input.empty()) {
            std::cerr << "input vector is empty\n";
            return 1;
        }
        float in_min = input[0], in_max = input[0];
        double in_sum = 0.0, in_sq = 0.0;
        for (float v : input) {
            in_min = std::min(in_min, v);
            in_max = std::max(in_max, v);
            in_sum += v;
            in_sq  += double(v) * double(v);
        }
        std::printf("# input size=%zu min=%.4g max=%.4g mean=%.4g rms=%.4g\n\n",
                    input.size(), in_min, in_max, in_sum / input.size(),
                    std::sqrt(in_sq / input.size()));

        // === Real block-0 weights — 3-way comparison ===
        // H = handwritten CPU (operator_backend's rows=shape[0] convention)
        // C = canonical CPU  (Session::runLinear — GGML cols=shape[0] convention)
        // M = Metal kernel   (called via the operator_backend dispatch path)
        // The pairwise cosines isolate kernel bugs from dispatch/convention bugs.
        std::printf("=== Real Q4_0 weights from block 0 (3-way: H=handwritten, C=canonical, M=metal) ===\n");
        std::printf("%-31s | %-13s | %-11s | %10s | %10s | %10s | %s\n",
                    "weight", "shape", "kernel",
                    "H<->C cos", "H<->M cos", "C<->M cos", "flag");
        std::printf("%s\n", std::string(120, '-').c_str());

        const std::vector<std::string> wnames = {
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight"
        };

        struct PerWeightDiag {
            std::string name;
            size_t shape0 = 0, shape1 = 0;
            std::vector<float> cm_buckets;
            std::vector<size_t> cm_worst_idx;
        };
        std::vector<PerWeightDiag> diags;

        for (const auto& name : wnames) {
            auto it = tensors.find(name);
            if (it == tensors.end()) {
                std::printf("%-31s | (not found in GGUF)\n", name.c_str());
                continue;
            }
            const auto& info = it->second;
            if (info.shape.size() != 2) {
                std::printf("%-31s | (shape rank != 2)\n", name.c_str());
                continue;
            }
            size_t shape0 = static_cast<size_t>(info.shape[0]);
            size_t shape1 = static_cast<size_t>(info.shape[1]);

            // Canonical GGML convention: cols=shape[0]=ne0 (input dim),
            // rows=shape[1]=ne1 (output dim). Matches Session::runLinear and
            // (post-Bug-B fix) MetalExecutionBackend::execute.
            size_t cols = shape0;
            size_t rows = shape1;
            bool transpose;
            if (cols == input.size())      transpose = false;
            else if (rows == input.size()) transpose = true;
            else {
                std::printf("%-31s | (neither cols nor rows match input %zu)\n",
                            name.c_str(), input.size());
                continue;
            }

            const auto& raw = session.tensorData(info);
            size_t row_stride = raw.size() / rows;

            auto handwritten = q4diag::cpuMatmulHandwritten(
                raw.data(), rows, cols, row_stride, qversion, input, transpose);

            std::vector<float> canonical;
            try {
                canonical = q4diag::cpuMatmulCanonical(session, name, input);
            } catch (const std::exception& e) {
                std::printf("%-31s | canonical (Session::runLinear) failed: %s\n",
                            name.c_str(), e.what());
                continue;
            }

            std::vector<float> metal;
            bool ok = transpose
                ? MetalExecutor::Instance().runMatMulQ4_0Transposed(
                      raw, input, rows, cols, row_stride, qversion, metal, nullptr)
                : MetalExecutor::Instance().runMatMulQ4_0(
                      raw, input, rows, cols, row_stride, qversion, metal, nullptr);
            if (!ok) {
                std::printf("%-31s | metal kernel returned ok=false\n", name.c_str());
                continue;
            }

            auto m_hc = q4diag::metrics(handwritten, canonical, 0);
            auto m_hm = q4diag::metrics(handwritten, metal, 0);
            auto m_cm = q4diag::metrics(canonical, metal, 8);

            char shape_buf[24];
            std::snprintf(shape_buf, sizeof(shape_buf), "[%zu,%zu]", shape0, shape1);
            const char* flag = q4diag::classifyDivergence(m_hc.cosine, m_hm.cosine, m_cm.cosine);
            std::printf("%-31s | %-13s | %-11s | %10.6f | %10.6f | %10.6f | %s\n",
                        name.c_str(), shape_buf,
                        transpose ? "transposed" : "non-trans",
                        m_hc.cosine, m_hm.cosine, m_cm.cosine, flag);

            PerWeightDiag d;
            d.name = name;
            d.shape0 = shape0;
            d.shape1 = shape1;
            d.cm_buckets = m_cm.bucket_mean_err;
            d.cm_worst_idx = m_cm.worst_idx;
            diags.push_back(std::move(d));
        }

        // === Bucket histogram of C↔M error across output dim (most actionable) ===
        std::printf("\n=== Mean |error| of C<->M (canonical vs metal) per output-dim bucket (16 slices) ===\n");
        std::printf("%-31s ", "weight");
        for (int b = 0; b < 16; ++b) std::printf(" B%-7d", b);
        std::printf("\n");
        for (const auto& d : diags) {
            std::printf("%-31s ", d.name.c_str());
            for (float v : d.cm_buckets) std::printf(" %-8.3g", v);
            std::printf("\n");
        }

        if (!do_stress) return 0;

        // === Synthetic stress: K=2048 fixed, vary N for both kernels ===
        std::printf("\n=== Synthetic stress (K=2048 fixed, vary N) ===\n");
        std::printf("%-13s | %-11s | %9s | %9s | %9s | %9s\n",
                    "shape", "kernel", "cosine", "max_abs", "mean_abs", "rms");
        std::printf("%s\n", std::string(80, '-').c_str());

        const std::vector<size_t> N_values = {64, 128, 256, 512, 1024, 2048, 4096, 8192};
        const size_t K = 2048;
        if (input.size() != K) {
            std::cerr << "(stress requires input size " << K
                      << " but have " << input.size() << ")\n";
            return 1;
        }
        std::vector<float> tmp_row;
        std::vector<uint8_t> tmp_quant;

        auto run_one = [&](size_t weight_rows, size_t weight_cols, bool transpose,
                           uint64_t shape_seed) {
            size_t row_stride = q4_0RowSize(weight_cols, qversion);
            std::vector<uint8_t> weights(weight_rows * row_stride);
            std::mt19937_64 rng(shape_seed);
            std::normal_distribution<float> dist(0.0f, 0.05f);
            tmp_row.assign(weight_cols, 0.0f);
            for (size_t r = 0; r < weight_rows; ++r) {
                for (auto& v : tmp_row) v = dist(rng);
                quantizeRowQ4_0(tmp_row.data(), weight_cols, qversion, tmp_quant);
                std::memcpy(weights.data() + r * row_stride, tmp_quant.data(),
                            std::min(tmp_quant.size(), row_stride));
            }
            auto cpu_out = q4diag::cpuMatmulHandwritten(weights.data(), weight_rows, weight_cols,
                                              row_stride, qversion, input, transpose);
            std::vector<float> metal_out;
            bool ok = transpose
                ? MetalExecutor::Instance().runMatMulQ4_0Transposed(
                      weights, input, weight_rows, weight_cols, row_stride, qversion,
                      metal_out, nullptr)
                : MetalExecutor::Instance().runMatMulQ4_0(
                      weights, input, weight_rows, weight_cols, row_stride, qversion,
                      metal_out, nullptr);
            char shape_buf[24];
            std::snprintf(shape_buf, sizeof(shape_buf), "[%zu,%zu]", weight_rows, weight_cols);
            if (!ok) {
                std::printf("%-13s | %-11s | (metal returned ok=false)\n",
                            shape_buf, transpose ? "transposed" : "non-trans");
                return;
            }
            auto m = q4diag::metrics(cpu_out, metal_out, 0);
            std::printf("%-13s | %-11s | %9.6f | %9.3e | %9.3e | %9.3e\n",
                        shape_buf, transpose ? "transposed" : "non-trans",
                        m.cosine, m.max_abs, m.mean_abs, m.rms);
        };

        for (size_t N : N_values) {
            // Non-transposed: weight is [N, K]; rows=N, cols=K. K is multiple of 32 ✓.
            run_one(/*weight_rows=*/N, /*weight_cols=*/K, /*transpose=*/false,
                    seed ^ (N * 0x9E3779B97F4A7C15ULL));
        }
        for (size_t N : N_values) {
            // Transposed: weight is [K, N]; rows=K, cols=N. Need N % 32 == 0.
            if (N % 32 != 0) continue;
            run_one(/*weight_rows=*/K, /*weight_cols=*/N, /*transpose=*/true,
                    seed ^ (N * 0xBF58476D1CE4E5B9ULL));
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test-matmul-q4: " << e.what() << "\n";
        return 1;
    }
}
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    auto args = mlc::util::parseArgs(argc, argv);
    
    // Handle help request
    if (args.help_requested) {
        if (args.command.empty() || args.command == "help") {
            mlc::util::printUsage(argv[0]);
        } else {
            mlc::util::printCommandHelp(args.command);
        }
        return 0;
    }
    
    // Handle commands
    if (args.command.empty()) {
        mlc::util::printUsage(argv[0]);
        return 1;
    }
    
    if (args.command == "inspect") {
        return handleInspectCommand(args);
    } else if (args.command == "meta") {
        return handleMetaCommand(args.arguments);
    } else if (args.command == "linear") {
        return handleLinearCommand(args.arguments);
    } else if (args.command == "embed") {
        return handleEmbedCommand(args.arguments);
    } else if (args.command == "plan") {
        return handlePlanCommand(args.arguments);
    } else if (args.command == "run") {
        return handleRunCommand(args);
    } else if (args.command == "decode") {
        return handleDecodeCommand(args);
    } else if (args.command == "chat") {
        return handleChatCommand(args.arguments);
    } else if (args.command == "chat-repl") {
        return handleChatReplCommand(args.arguments);
    } else if (args.command == "chat-llama") {
        return handleChatLlamaCommand(args.arguments);
    } else if (args.command == "tokenize") {
        return handleTokenizeCommand(args.arguments);
    } else if (args.command == "detokenize") {
        return handleDetokenizeCommand(args.arguments);
    } else if (args.command == "compare") {
        return handleCompareCommand(args.arguments);
    } else if (args.command == "test-matmul-q4") {
        return handleTestMatmulQ4Command(args.arguments);
    } else if (args.command == "capabilities") {
        return handleCapabilitiesCommand();
    } else {
        std::cerr << "Unknown command: " << args.command << "\n";
        std::cerr << "Use 'mlc help' for available commands.\n";
        return 1;
    }
}
