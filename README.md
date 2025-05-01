# Colab Demo: https://colab.research.google.com/drive/1Or2jfyb5BUneN5wUsMMYCqCl3esFDi6f#scrollTo=6Zo7MWQ1eBDq

# Cuda-C Neural Network Accelerator — Technical Writeup

---

## 1. Overall Architecture

- **Matrix Abstraction**  
  - `Matrix` class manages host (CPU) and device (GPU) buffers  
  - Encapsulates `cudaMalloc`, `cudaFree`, and `cudaMemcpy`  
  - Provides row‐major storage and simple indexing

- **Layer Stack**  
  - Each fully-connected layer holds:
    - **Weights** and **biases** in device memory  
    - Activation buffers for forward and backward passes  
    - Gradient buffers for weight, bias, and input derivatives

- **Training Loop**  
  1. **Forward Pass**: propagate inputs through layers to compute logits  
  2. **Loss**: compute softmax-cross-entropy via cuDNN  
  3. **Backward Pass**: backpropagate gradients through each layer  
  4. **Update**: apply Adam optimizer on GPU to adjust parameters

---

## 2. Data Flow & Memory Layout

1. **Host Data Load**  
   - CSV reader in C++ loads features & labels into host arrays.  
2. **Host→Device Transfer**  
   - Entire minibatch copied once per iteration via `cudaMemcpy`.  
3. **Layer Execution**  
   - **GEMM**: `cublasSgemm` for `A×W` (forward) and `dA×Wᵀ` (backward)  
   - **Activation**: custom kernel applies ReLU (or others) element-wise  
4. **Softmax & Loss**  
   - `cudnnSoftmaxForward` computes probabilities efficiently  
   - Custom cross-entropy backward kernel computes gradients w.r.t. logits  
5. **Optimizer**  
   - Custom Adam kernel updates parameters and running moments in place

---

## 3. Custom CUDA Kernels

### 3.1 Activation Functions

```cpp
__global__ void relu_forward(float* x, int N) { 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) x[i] = fmaxf(0.0f, x[i]);
}
__global__ void relu_backward(float* grad, float* inp, int N) { 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) grad[i] *= (inp[i] > 0.0f);
}
```
- **Mapping**: one thread per element  
- **Access**: coalesced reads/writes for throughput  

### 3.2 Adam Optimizer

```cpp
__global__ void adam_update(
    float* params, float* grads,
    float* m, float* v,
    float lr, int t, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        float g = grads[i];
        m[i] = 0.9f * m[i] + 0.1f * g;
        v[i] = 0.999f * v[i] + 0.001f * (g * g);
        float m_hat = m[i] / (1.0f - powf(0.9f, t));
        float v_hat = v[i] / (1.0f - powf(0.999f, t));
        params[i] -= lr * m_hat / (sqrtf(v_hat) + 1e-8f);
    }
}
```
- **Moment Estimates**: bias-corrected in-kernel  
- **Update Rule**: follows standard Adam equations  

---

## 4. cuBLAS & cuDNN Integration

- **cuBLAS**  
  - `cublasSgemm` for dense matrix multiplies  
  - Transposition flags manage forward vs. backward GEMMs  

- **cuDNN**  
  - `cudnnSoftmaxForward` for numerically stable softmax  
  - `cudnnSoftmaxBackward` for cross-entropy gradient  

---

## 5. Synchronization & Streams

- **Default Stream**: sequential execution per layer  
- **`cudaDeviceSynchronize()`** after each kernel ensures correctness before CPU operations (e.g., bias addition)

---

## 6. Performance Considerations

- **Batch Size**: chosen to maximize GPU utilization  
- **Buffer Reuse**: allocations done once, reused across epochs  
- **Kernel Occupancy**: blocks of 256 threads to fill SMs  
- **Future Optimizations**:  
  - Fuse GEMM + activation  
  - Offload bias addition to CUDA  
  - Employ mixed-precision tensor cores  
