import os
import subprocess
import sys

if len(sys.argv) != 2:
    print("Usage: python script.py <cpp_file>")
    sys.exit(1)

cpp_file = sys.argv[1]

# Extract the base filename without extension
base_filename = os.path.splitext(cpp_file)[0]

# Create run_files directory if it doesn't exist
run_files_dir = 'run_files'
os.makedirs(run_files_dir, exist_ok=True)

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
run_command = f"./{cuda_file}"
subprocess.run(run_command, shell=True, check=True)