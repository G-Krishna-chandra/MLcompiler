#include "cli_helpers.hpp"
#include <iostream>
#include <algorithm>

namespace mlc {
namespace util {

ParsedArgs parseArgs(int argc, char* argv[]) {
    ParsedArgs args;
    
    if (argc < 2) {
        args.help_requested = true;
        return args;
    }
    
    // First argument is the command
    args.command = argv[1];
    
    // Check for help flags
    if (args.command == "--help" || args.command == "-h" || args.command == "help") {
        args.help_requested = true;
        return args;
    }
    
    // Remaining arguments
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            args.help_requested = true;
            break;
        } else if (arg == "--verbose" || arg == "-v") {
            args.verbose = true;
        } else if (arg == "--dump-header") {
            args.dump_header = true;
        } else if (arg == "--dump-tensors") {
            args.dump_tensors = true;
        } else if (arg == "--dump-one" && i + 1 < argc) {
            args.dump_one = argv[++i];
        } else if (arg == "--hex" && i + 1 < argc) {
            args.hex_bytes = std::stoi(argv[++i]);
        } else {
            args.arguments.push_back(arg);
        }
    }
    
    return args;
}

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <command> [arguments]\n\n";
    std::cout << "Commands:\n";
    std::cout << "  inspect <gguf_path>    Inspect a GGUF model file\n";
    std::cout << "  meta <gguf_path>       Print GGUF metadata and tensor shapes\n";
    std::cout << "  linear <gguf_path> <tensor> <values>\n";
    std::cout << "                       Run a single linear layer with comma-separated input values\n";
    std::cout << "  embed <gguf_path> <tensor> <token_id>\n";
    std::cout << "                       Fetch an embedding vector for a specific token id\n";
    std::cout << "  plan <gguf_path>       Build a high-level execution plan for the model\n";
    std::cout << "  run [options] <gguf_path> [token]\n";
    std::cout << "                       Dry-run the execution plan, embeddings/logits, and schedule\n";
    std::cout << "  capabilities          Show built-in runtime/Metal capabilities\n";
    std::cout << "  help                   Show this help message\n";
    std::cout << "\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " inspect model.gguf\n";
    std::cout << "  " << program_name << " meta model.gguf\n";
    std::cout << "  " << program_name << " help\n";
}

void printCommandHelp(const std::string& command) {
    if (command == "meta") {
        std::cout << "Command: meta\n";
        std::cout << "  Print GGUF metadata (arch, n_vocab, n_embd) and tensor information.\n\n";
        std::cout << "Usage: mlc meta <gguf_path>\n\n";
        std::cout << "Arguments:\n";
        std::cout << "  gguf_path    Path to the GGUF model file\n\n";
        std::cout << "Example:\n";
        std::cout << "  mlc meta model.gguf\n";
    } else if (command == "inspect") {
        std::cout << "Command: inspect\n";
        std::cout << "  Inspect a GGUF model file and display tensor information.\n\n";
        std::cout << "Usage: mlc inspect [options] <gguf_path>\n\n";
        std::cout << "Options:\n";
        std::cout << "  --verbose, -v         Enable verbose output with detailed tensor metadata\n";
        std::cout << "  --dump-header         Print all GGUF KV metadata entries\n";
        std::cout << "  --dump-tensors        Print detailed tensor information\n";
        std::cout << "  --dump-one <name>     Print metadata for specific tensor\n";
        std::cout << "  --hex <N>             Print first N bytes of tensor in hex (use with --dump-one)\n\n";
        std::cout << "Examples:\n";
        std::cout << "  mlc inspect model.gguf\n";
        std::cout << "  mlc inspect --verbose model.gguf\n";
        std::cout << "  mlc inspect --dump-header model.gguf\n";
        std::cout << "  mlc inspect --dump-tensors model.gguf\n";
        std::cout << "  mlc inspect --dump-one output.weight --hex 64 model.gguf\n";
    } else if (command == "linear") {
        std::cout << "Command: linear\n";
        std::cout << "  Execute a single GGUF tensor as a linear layer on CPU.\n\n";
        std::cout << "Usage: mlc linear <gguf_path> <tensor_name> <comma-separated-input>\n\n";
        std::cout << "Arguments:\n";
        std::cout << "  gguf_path      Path to the GGUF model file\n";
        std::cout << "  tensor_name    Name of a 2D F32 weight tensor in the file\n";
        std::cout << "  input          Comma-separated list of floating point values (size must match tensor columns)\n\n";
        std::cout << "Example:\n";
        std::cout << "  mlc linear model.gguf output.weight 0.1,0.2,0.3\n";
    } else if (command == "embed") {
        std::cout << "Command: embed\n";
        std::cout << "  Fetch a single embedding vector from a 2D F32 tensor.\n\n";
        std::cout << "Usage: mlc embed <gguf_path> <tensor_name> <token_id>\n\n";
        std::cout << "Arguments:\n";
        std::cout << "  gguf_path      Path to the GGUF model file\n";
        std::cout << "  tensor_name    Name of the embedding tensor (e.g., tok_embeddings.weight)\n";
        std::cout << "  token_id       Integer row index within the tensor\n\n";
        std::cout << "Example:\n";
        std::cout << "  mlc embed model.gguf tok_embeddings.weight 42\n";
    } else if (command == "plan") {
        std::cout << "Command: plan\n";
        std::cout << "  Build and display the compiler's execution graph for the model.\n\n";
        std::cout << "Usage: mlc plan <gguf_path>\n\n";
        std::cout << "Arguments:\n";
        std::cout << "  gguf_path      Path to the GGUF model file\n\n";
        std::cout << "Example:\n";
        std::cout << "  mlc plan model.gguf\n";
    } else if (command == "run") {
        std::cout << "Command: run\n";
        std::cout << "  Dry-run the execution graph: show schedule, embeddings, optional logits,\n";
        std::cout << "  and (with --simulate) walk the execution graph without kernels.\n\n";
        std::cout << "Usage: mlc run [options] <gguf_path> [token_id]\n\n";
        std::cout << "Options:\n";
        std::cout << "  --token <id>        Token id to fetch (default 0)\n";
        std::cout << "  --preview <N>       Number of values to print from embeddings/logits (default 8)\n";
        std::cout << "  --no-logits         Skip attempting the output projection\n";
        std::cout << "  --simulate          Walk the execution graph and print each node\n";
        std::cout << "  --simulate-limit N  Stop after visiting N nodes during simulation\n";
        std::cout << "  --verbose           Also dump the textual execution plan\n\n";
        std::cout << "Examples:\n";
        std::cout << "  mlc run model.gguf 1\n";
        std::cout << "  mlc run --token 42 --preview 4 model.gguf\n";
    } else if (command == "capabilities") {
        std::cout << "Command: capabilities\n";
        std::cout << "  Display the compiled runtime backends and Metal kernel availability.\n\n";
        std::cout << "Usage: mlc capabilities\n\n";
        std::cout << "The output lists whether the binary includes Metal kernels for\n";
        std::cout << "feed-forward, residual add, and RMS norm, and whether a Metal device\n";
        std::cout << "is available at runtime. When kernels are missing or the device is\n";
        std::cout << "unavailable, the runtime automatically falls back to CPU/Accelerate.\n";
    } else {
        std::cout << "Unknown command: " << command << "\n";
        std::cout << "Use 'mlc help' for available commands.\n";
    }
}

} // namespace util
} // namespace mlc
