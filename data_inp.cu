#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

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
    const int num_samples = 35700;
    const int num_features = 784;


    // Allocate arrays
    Matrix inputs(num_samples, num_features);
    Matrix y_true(1, num_samples);


    readData(filename, inputs.data, y_true.data, num_samples, num_features);

    // if you want dummy data:
    // inputs.init_random();
    // y_true.init_int(1);

    int layer1_out = 64;
    int layer2_out = 64;

    Layer_Dense layer1(num_samples, num_features, layer1_out);
    Activation_Relu act1(num_samples, layer1_out);

    Layer_Dense layer2(num_samples, layer1_out, layer2_out);
    Activation_Relu act2(num_samples, layer2_out);

    Layer_Dense layer3(num_samples, layer2_out, 10);

    Softmax_CE_Loss smceLoss(num_samples, 10);
    Optimizer_Adam optimizer(0.16, 2e-7);

    float total_time = 0;

    {
        Timer timer(total_time);
        for (int epoch = 0; epoch < 1000; epoch++) {
            // Forward pass
            layer1.forward(inputs);
            act1.forward(layer1.output);
            layer2.forward(act1.output);
            act2.forward(layer2.output);
            layer3.forward(act2.output);
            smceLoss.forward(layer3.output);

            // Backward pass
            smceLoss.backward(smceLoss.output, y_true);
            layer3.backward(smceLoss.dinputs);
            act2.backward(layer3.dinputs);
            layer2.backward(act2.dinputs);
            act1.backward(layer2.dinputs);
            layer1.backward(act1.dinputs);

            // Optimize
            optimizer.pre_update_params();
            optimizer.update_params(layer1);
            optimizer.update_params(layer2);
            optimizer.update_params(layer3);
            optimizer.post_update_params();

        // if ((epoch + 1) % 50 == 0 || epoch == 0)
        //     std::cout << "epoch: " << epoch << " loss: " << smceLoss.calc_loss_cpu(smceLoss.output, y_true) << " accuracy: " << smceLoss.calc_acc_cpu(smceLoss.output, y_true) << std::endl;
        }
        cudaDeviceSynchronize();
    }



    std::cout << std::fixed << std::setprecision(10);

    // Print the total training CPU time
    std::cout << "total_training_CPU_time: " << total_time << " seconds" << std::endl;







    // This code will read the test data and calculate the accuracy of the model on unseen data
    const int test_num_samples = 6300;
    // num_features = 784;

    // init with the same dims as original batch so it will fit in the mmodel
    Matrix test_inputs(num_samples, num_features);
    Matrix test_y_true(1, num_samples);

    readData("test_data.txt", test_inputs.data, test_y_true.data, test_num_samples, num_features);

    // the rest of the data should be filled with zeroes

    layer1.forward(test_inputs);
    act1.forward(layer1.output);

    layer2.forward(act1.output);
    act2.forward(layer2.output);

    layer3.forward(act2.output);
    smceLoss.forward(layer3.output);


    // set layer2.smceLoss.rows to 6300. this is false but we will exploit our accuracy fn this way
    smceLoss.output.rows = 6300;
    cout<< "TESTset accuracy: " << smceLoss.calc_acc_cpu(smceLoss.output, test_y_true, false) <<endl;

    // END BLOCK OF CODE TO TEST ACC





    CHECK_CUBLAS(cublasDestroy(global_cublas_handle));
    CHECK_CUDNN(cudnnDestroy(global_cudnn_handle));
    return 0;

}

