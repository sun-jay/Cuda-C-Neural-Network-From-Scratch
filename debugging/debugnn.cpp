#include "nn.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>



void printCUDAMemoryInfo() {
    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemGetInfo failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    setprecision(10);

    std::cout << "Free memory: " << free_mem / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Total memory: " << total_mem / (1024.0 * 1024.0) << " MB" << std::endl;
}

int main(){
    CHECK_CUBLAS(cublasCreate(&global_cublas_handle));
    CHECK_CUDNN(cudnnCreate(&global_cudnn_handle));

    // hts(n_neurons, n_inputs), biases(n_neurons, 1),
    //       output(n_neurons, batch_size), dweights
    int batch_size = 34000;
    int n_inputs = 784;
    int n_neurons = 64;

    // int batch_size = 3;
    // int n_inputs = 3;
    // int n_neurons = 3;

    // rows, cols

    Matrix inputs(batch_size, n_inputs);
    inputs.init_random();

    Matrix y_true(1, batch_size);
    y_true.init_int(0);

    
    Layer_Dense layer1(batch_size, n_inputs, n_neurons);
    Activation_Relu act1(batch_size, n_neurons);
    Softmax_CE_Loss smceLoss(batch_size, n_neurons);
    Optimizer_Adam optimizer(0.5, 5e-7);
    

    for(int epoch = 0;  epoch <100; epoch ++){
    
    {
    if (epoch%10 == 1)
    CudaTimer("fwd bkwd");

    // inputs.print();
    
    layer1.forward(inputs);
    act1.forward(layer1.output);

    smceLoss.forward(act1.output);
    // CHECK_LAST_CUDA_ERROR();
    // printCUDAMemoryInfo();



    
    smceLoss.backward(smceLoss.output, y_true);
    // cout<<"passed smceLoss fwd and b"<<endl;

    // CHECK_LAST_CUDA_ERROR();
    // printCUDAMemoryInfo();
    act1.backward(smceLoss.dinputs);
    layer1.backward(act1.dinputs);


    optimizer.pre_update_params();
    optimizer.update_params(layer1);
    optimizer.post_update_params();
    
    if (epoch%10 == 1)
    cout<<"epoch: " << epoch << "loss: " << smceLoss.calc_loss_cpu(smceLoss.output, y_true) <<endl;

    }

    }

}









// #include "nn.h"
// #include <cudnn.h>
// #include <iostream>
// #include <cuda_runtime.h>

// int main() {
//     cudnnHandle_t lib;
//     CHECK_CUDNN(cudnnCreate(&lib));

//     int num_samples = 3; // Number of samples
//     int count = 10;      // Number of elements per sample
//     size_t size = num_samples * count * sizeof(float);

//     // Initialize examples with two samples
//     float examples[] = {
//         95.094505f, -600.288879f, 85.621284f, 72.220154f, 70.099487f, 333.734470f, 69.538422f, 333.705490f, 20.752966f, 333.020088f,
//         60.123456f, -450.123456f, 50.654321f, 45.321654f, 40.987654f, 250.654321f, 35.876543f, 255.654321f, 10.123456f, 100000.123456f,
//         60.123456f, 100000.123456f, 50.654321f, 45.321654f, 40.987654f, 250.654321f, 35.876543f, 255.654321f, 10.123456f, 100000.123456f
//     };
    
//     Matrix e(num_samples,count);

//     for (int i = 0; i<num_samples*count;i++ ){
//         e.data[i] = examples[i];
//     }

//     float* cexamples;
//     CHECK_CUDA(cudaMalloc(&cexamples, size));
//     CHECK_CUDA(cudaMemcpy(cexamples, examples, size, cudaMemcpyHostToDevice));

//     cudnnTensorDescriptor_t tExamples;
//     CHECK_CUDNN(cudnnCreateTensorDescriptor(&tExamples));
//     CHECK_CUDNN(cudnnSetTensor4dDescriptor(tExamples, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_samples, count,1, 1));

//     float one = 1;
//     float zero = 0;

//     CHECK_CUDNN(cudnnSoftmaxForward(lib, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &one, tExamples, e.data, &zero, tExamples, cexamples));

//     CHECK_CUDA(cudaMemcpy(examples, cexamples, size, cudaMemcpyDeviceToHost));

//     // Print results for each sample
//     for (int sample = 0; sample < num_samples; ++sample) {
//         std::cout << "Sample " << sample + 1 << ":" << std::endl;
//         for (int i = 0; i < count; ++i) {
//             std::cout << examples[sample * count + i] << " ";
//         }
//         std::cout << std::endl;
//     }

//     // Cleanup
//     CHECK_CUDA(cudaFree(cexamples));
//     CHECK_CUDNN(cudnnDestroyTensorDescriptor(tExamples));
//     CHECK_CUDNN(cudnnDestroy(lib));

//     return 0;
// }
