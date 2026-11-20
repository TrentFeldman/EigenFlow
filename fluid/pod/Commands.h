// fluid_pod subcommands. Each takes the arguments after the subcommand name and returns an exit code
// (errors are thrown as pod::Error and reported by run_cli).
#pragma once
#include <string>
#include <vector>

namespace pod {

int run_cli(const std::vector<std::string>& args);  // args exclude the program name

int cmd_info(const std::vector<std::string>& args);
int cmd_convert(const std::vector<std::string>& args);
int cmd_pod(const std::vector<std::string>& args);
int cmd_reconstruct(const std::vector<std::string>& args);
int cmd_export(const std::vector<std::string>& args);

}  // namespace pod
