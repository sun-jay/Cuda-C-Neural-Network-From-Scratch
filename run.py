import os
import subprocess
import sys

if len(sys.argv) != 2:
    print("Usage: python script.py <cpp_file>")
    sys.exit(1)

cpp_file = sys.argv[1]

# Extract the base filename without extension
base_filename = os.path.splitext(cpp_file)[0]

# Step 1: Copy and rename the provided C++ file to .cu
cu_file = base_filename + '.cu'

with open(cpp_file, "r") as src_file:
    with open(cu_file, "w") as dst_file:
        dst_file.write(src_file.read())

# Step 2: Compile the CUDA file
compile_command = f"nvcc {cu_file} -o {base_filename}_cuda -lcublas -lcudnn"
subprocess.run(compile_command, shell=True, check=True)

# Step 3: Run the compiled CUDA executable
run_command = f"./{base_filename}_cuda"
subprocess.run(run_command, shell=True, check=True)
