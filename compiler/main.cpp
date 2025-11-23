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
#include "util/cli_helpers.hpp"

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
        std::stringstream ss(text);
        std::string token;
        while (std::getline(ss, token, ',')) {
            std::string cleaned = trim(token);
            if (cleaned.empty()) continue;
            values.push_back(std::stof(cleaned));
        }
        if (values.empty()) {
            throw std::runtime_error("Input list must contain at least one value");
        }
        return values;
    }

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
        const std::string& input_values = args[2];

        try {
            auto inputs = parseFloatList(input_values);
            mlc::runtime::Session session(gguf_path);
            auto outputs = session.runLinear(tensor_name, inputs);

            std::cout << "Linear output (" << outputs.size() << " values):\n";
            for (size_t i = 0; i < outputs.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << outputs[i];
            }
            std::cout << "\n";
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
    } else if (args.command == "capabilities") {
        return handleCapabilitiesCommand();
    } else {
        std::cerr << "Unknown command: " << args.command << "\n";
        std::cerr << "Use 'mlc help' for available commands.\n";
        return 1;
    }
}
