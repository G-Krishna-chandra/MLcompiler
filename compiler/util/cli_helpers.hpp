#pragma once

#include <string>
#include <vector>

namespace mlc {
namespace util {

struct ParsedArgs {
    std::string command;
    std::vector<std::string> arguments;
    bool help_requested = false;
    bool verbose = false;
    bool dump_header = false;
    bool dump_tensors = false;
    std::string dump_one;  // Tensor name to dump
    int hex_bytes = 0;  // Number of bytes to dump in hex (0 = disabled)
};

// Parse command line arguments
ParsedArgs parseArgs(int argc, char* argv[]);

// Print usage information
void printUsage(const char* program_name);

// Print help for a specific command
void printCommandHelp(const std::string& command);

} // namespace util
} // namespace mlc


