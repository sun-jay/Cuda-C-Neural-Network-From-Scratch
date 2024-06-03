import os
import subprocess
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Compile and run a CUDA file.")
parser.add_argument("cpp_file", help="The C++ file to compile and run.")
parser.add_argument("--profile", action="store_true", help="Run with nvprof.")

# Parse the arguments
args = parser.parse_args()

cpp_file = args.cpp_file

# Extract the base filename without extension
base_filename = os.path.splitext(cpp_file)[0]

# Create run_files directory if it doesn't exist
run_files_dir = 'run_files'
os.makedirs(run_files_dir, exist_ok=True)

# Copy nn.h into run_files using a command
copy_command = f"cp nn.h {run_files_dir}"
subprocess.run(copy_command, shell=True, check=True)

# Step 1: Copy and rename the provided C++ file to .cu
cu_file = os.path.join(run_files_dir, base_filename + '.cu')

with open(cpp_file, "r") as src_file:
    with open(cu_file, "w") as dst_file:
        dst_file.write(src_file.read())

# Step 2: Compile the CUDA file
cuda_file = os.path.join(run_files_dir, base_filename + '_cuda')
compile_command = f"nvcc {cu_file} -o {cuda_file} -lcublas -lcudnn"
subprocess.run(compile_command, shell=True, check=True)

# Step 3: Run the compiled CUDA executable
run_command = f"./{base_filename}_cuda"
if args.profile:
    run_command = f"nvprof {run_command}"
subprocess.run(run_command, shell=True, check=True)