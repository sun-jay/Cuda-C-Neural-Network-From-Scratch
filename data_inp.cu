#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

#include <thread>
#include <chrono>

#include "nn.h"

void readData(const std::string& filename, float* X, float* Y, int num_samples, int num_features) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    std::string line;
    int sample_idx = 0;

    while (std::getline(file, line) && sample_idx < num_samples) {
        std::stringstream ss(line);
        std::string value;
        int feature_idx = 0;

        while (std::getline(ss, value, ',')) {
            if (feature_idx < num_features) {
                X[sample_idx * num_features + feature_idx] = std::stof(value);
                ++feature_idx;
            }
        }

        if (std::getline(file, line)) {
            Y[sample_idx] = std::stoi(line);
            ++sample_idx;
        }
    }

    file.close();
}

int main() {
    CHECK_CUBLAS(cublasCreate(&global_cublas_handle));
    CHECK_CUDNN(cudnnCreate(&global_cudnn_handle));

    const std::string filename = "data.txt";
    const int num_samples = 35700*2;
    const int num_features = 784*2;

    // const int num_samples = 13;
    // const int num_features = 13;

    

    // Allocate arrays
    Matrix inputs(num_samples, num_features);
    Matrix y_true(1, num_samples);

    inputs.init_random();
    y_true.init_int(1);

    // readData(filename, inputs.data, y_true.data, num_samples, num_features);






    
    inputs.init_random();
    y_true.init_int(1);
    
    
    // for (int i = 35700-10; i< 35700;i++ ){
    //     cout << y_true.data[i] <<endl;
    // }
    // return 0;

    Layer_Dense layer1(num_samples, num_features, 64);
    Activation_Relu act1(num_samples, 64);

    Layer_Dense layer2(num_samples, 64, 10);
    // Activation_Relu act2(num_samples, 10);

    Softmax_CE_Loss smceLoss(num_samples, 10);
    Optimizer_Adam optimizer(0.001, 5e-4);

    float total_time = 0;
    

    for(int epoch = 0;  epoch <2; epoch ++){
        
        {
        StdTimer("fwd bkwd", total_time, epoch, 1); //timing block

        std::this_thread::sleep_for(std::chrono::seconds(1));
    
        layer1.forward(inputs);
        act1.forward(layer1.output);

        layer2.forward(act1.output);
        // act2.forward(layer2.output);

        smceLoss.forward(layer2.output); 

        // Backward pass
        smceLoss.backward(smceLoss.output, y_true);
        // act2.backward(smceLoss.dinputs);
        layer2.backward(smceLoss.dinputs); 

        act1.backward(layer2.dinputs);
        layer1.backward(act1.dinputs);

        // Update parameters
        optimizer.pre_update_params();
        optimizer.update_params(layer1);
        optimizer.update_params(layer2);
        optimizer.post_update_params();
        }  //end timing block

    if (epoch%10 == 0)
    cout<<"epoch: " << epoch << " loss: " << smceLoss.calc_loss_cpu(smceLoss.output, y_true) <<endl;
    cout<<"epoch: " << epoch << " accuracy: " << smceLoss.calc_acc_cpu(smceLoss.output, y_true) <<endl;
    }

    cout << "total_gpu_time: " << total_time <<endl;
    

    


    
    CHECK_CUBLAS(cublasDestroy(global_cublas_handle));
    CHECK_CUDNN(cudnnDestroy(global_cudnn_handle));
    return 0;
}












    // int batch_size = num_samples;
    // int n_inputs = num_features;
    // int n_neurons = 64;


    // Matrix inputs(batch_size, n_inputs);
    // inputs.init_random();

    // Matrix y_true(1, batch_size);
    // y_true.init_int(1);

    
    // Layer_Dense layer1(batch_size, n_inputs, n_neurons);
    // Activation_Relu act1(batch_size, n_neurons);
    // Softmax_CE_Loss smceLoss(batch_size, n_neurons);
    // Optimizer_Adam optimizer(0.05, 5e-7);
    

    // for(int epoch = 0;  epoch <100; epoch ++){
    
    // {CudaTimer("fwd bkwd");
    
    // layer1.forward(inputs);
    // act1.forward(layer1.output);
    // smceLoss.forward(act1.output);
    
    // smceLoss.backward(smceLoss.output, y_true);
    // act1.backward(smceLoss.dinputs);
    // layer1.backward(act1.dinputs);

    // optimizer.pre_update_params();
    // optimizer.update_params(layer1);
    // optimizer.post_update_params();

    // cout<<"epoch: " << epoch << "loss: " << smceLoss.calc_loss_cpu(smceLoss.output, y_true) <<endl;

    // }

    // }
