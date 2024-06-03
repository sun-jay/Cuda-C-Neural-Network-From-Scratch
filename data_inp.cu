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
    const int num_samples = 35700;
    const int num_features = 784;

    // const int num_samples = 13;
    // const int num_features = 13;

    

    // Allocate arrays
    Matrix inputs(num_samples, num_features);
    Matrix y_true(1, num_samples);


    readData(filename, inputs.data, y_true.data, num_samples, num_features);
    
    // inputs.init_random();
    // y_true.init_int(1);


    
    
    // for (int i = 35700-10; i< 35700;i++ ){
    //     cout << y_true.data[i] <<endl;
    // }
    // return 0;

    int layer1_out = 64;

    Layer_Dense layer1(num_samples, num_features, layer1_out);
    Activation_Relu act1(num_samples, layer1_out);

    Layer_Dense layer2(num_samples, layer1_out, 10);
    // Activation_Relu act2(num_samples, 10);

    Softmax_CE_Loss smceLoss(num_samples, 10);
    Optimizer_Adam optimizer(0.4, 2e-7);

    float total_time = 0;
    float fwd_L1_time = 0;
    float fwd_RELU_time = 0;
    float fwd_L2_time = 0;
    float fwd_smceloss_time = 0;

    float bkwd_smceLoss_time = 0;
    float bkwd_L2_time = 0;
    float bkwd_RELU_time = 0;
    float bkwd_L1_time = 0;

    float optim_time = 0;

    // std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTime;
    // m_StartTime = std::chrono::high_resolution_clock::now();

    // for (int epoch = 0; epoch < 1000; epoch++) {

        
    //         // Timer timer(total_time);
    //         // Forward pass
    //         std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTimeI;
    //         m_StartTimeI = std::chrono::high_resolution_clock::now();
            
    //             // Timer timer(fwd_L1_time);
    //             layer1.forward(inputs);
            
            
    //             // Timer timer(fwd_RELU_time);
    //             act1.forward(layer1.output);
            
            
    //             // Timer timer(fwd_L2_time);
    //             layer2.forward(act1.output);
            
            
    //             // Timer timer(fwd_smceloss_time);
    //             smceLoss.forward(layer2.output);
            

    //         // Backward pass
            
    //             // Timer timer(bkwd_smceLoss_time);
    //             smceLoss.backward(smceLoss.output, y_true);
            
    //         // act2.backward(smceLoss.dinputs);
            
    //             // Timer timer(bkwd_L2_time);
    //             layer2.backward(smceLoss.dinputs);
            
            
    //             // Timer timer(bkwd_RELU_time);
    //             act1.backward(layer2.dinputs);
            
            
    //             // Timer timer(bkwd_L1_time);
    //             layer1.backward(act1.dinputs);
            
    //         // Update parameters
            
    //             // Timer timer(optim_time);
    //             optimizer.pre_update_params();
    //             optimizer.update_params(layer1);
    //             optimizer.update_params(layer2);
    //             optimizer.post_update_params();

    //             std::chrono::time_point<std::chrono::high_resolution_clock> m_EndTimeI;
    //             m_EndTimeI = std::chrono::high_resolution_clock::now();

    //             total_time += chrono::duration_cast<std::chrono::milliseconds>(m_EndTimeI - m_StartTimeI).count();
            
        
    // }

    // std::chrono::time_point<std::chrono::high_resolution_clock> m_EndTime;
    // m_EndTime = std::chrono::high_resolution_clock::now();

    // cout<< "Total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(m_EndTime - m_StartTime).count()/1000.0<<endl;
    // cout<< "Total time agregated: " << total_time/1000.0<<endl;





        { Timer timer(total_time);

    for (int epoch = 0; epoch < 1000; epoch++) {

            // Forward pass
            {
                Timer timer(fwd_L1_time);
                layer1.forward(inputs);
            }
            {
                Timer timer(fwd_RELU_time);
                act1.forward(layer1.output);
            }
            {
                Timer timer(fwd_L2_time);
                layer2.forward(act1.output);
            }
            {
                Timer timer(fwd_smceloss_time);
                smceLoss.forward(layer2.output);
            }

            // Backward pass
            {
                Timer timer(bkwd_smceLoss_time);
                smceLoss.backward(smceLoss.output, y_true);
            }
            // act2.backward(smceLoss.dinputs);
            {
                Timer timer(bkwd_L2_time);
                layer2.backward(smceLoss.dinputs);
            }
            {
                Timer timer(bkwd_RELU_time);
                act1.backward(layer2.dinputs);
            }
            {
                Timer timer(bkwd_L1_time);
                layer1.backward(act1.dinputs);
            }

            // Update parameters
            {
                Timer timer(optim_time);
                optimizer.pre_update_params();
                optimizer.update_params(layer1);
                optimizer.update_params(layer2);
                optimizer.post_update_params();
            }

        // if ((epoch + 1) % 50 == 0 || epoch == 0)
        //     std::cout << "epoch: " << epoch << " loss: " << smceLoss.calc_loss_cpu(smceLoss.output, y_true) << " accuracy: " << smceLoss.calc_acc_cpu(smceLoss.output, y_true) << std::endl;
    }

        }





    std::cout << std::fixed << std::setprecision(10);

    // Print the total training GPU time
    std::cout << "total_training_gpu_time: " << total_time << " seconds" << std::endl;

    // Calculate the total segment time
    float total_segment_time = fwd_L1_time + fwd_RELU_time + fwd_L2_time + fwd_smceloss_time +
                               bkwd_smceLoss_time + bkwd_L2_time + bkwd_RELU_time + bkwd_L1_time +
                               optim_time;

    // Print the total segment time and percentage breakdowns in a nicely formatted chart
    std::cout << "Total Time Breakdown:" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "| Segment               | Time (seconds) | Percentage  |" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "| Forward Layer 1       | " << std::setw(14) << fwd_L1_time << " | " << std::setw(10) << (fwd_L1_time / total_segment_time) * 100.0 << " % |" << std::endl;
    std::cout << "| Forward ReLU          | " << std::setw(14) << fwd_RELU_time << " | " << std::setw(10) << (fwd_RELU_time / total_segment_time) * 100.0 << " % |" << std::endl;
    std::cout << "| Forward Layer 2       | " << std::setw(14) << fwd_L2_time << " | " << std::setw(10) << (fwd_L2_time / total_segment_time) * 100.0 << " % |" << std::endl;
    std::cout << "| Forward SMCELoss      | " << std::setw(14) << fwd_smceloss_time << " | " << std::setw(10) << (fwd_smceloss_time / total_segment_time) * 100.0 << " % |" << std::endl;
    std::cout << "| Backward SMCELoss     | " << std::setw(14) << bkwd_smceLoss_time << " | " << std::setw(10) << (bkwd_smceLoss_time / total_segment_time) * 100.0 << " % |" << std::endl;
    std::cout << "| Backward Layer 2      | " << std::setw(14) << bkwd_L2_time << " | " << std::setw(10) << (bkwd_L2_time / total_segment_time) * 100.0 << " % |" << std::endl;
    std::cout << "| Backward ReLU         | " << std::setw(14) << bkwd_RELU_time << " | " << std::setw(10) << (bkwd_RELU_time / total_segment_time) * 100.0 << " % |" << std::endl;
    std::cout << "| Backward Layer 1      | " << std::setw(14) << bkwd_L1_time << " | " << std::setw(10) << (bkwd_L1_time / total_segment_time) * 100.0 << " % |" << std::endl;
    std::cout << "| Optimization          | " << std::setw(14) << optim_time << " | " << std::setw(10) << (optim_time / total_segment_time) * 100.0 << " % |" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "| Total Segment Time    | " << std::setw(14) << total_segment_time << " | " << std::setw(10) << 100.0 << " % |" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;







    // BLOCK OF CODE TO TEST ACC
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
    smceLoss.forward(layer2.output); 

    // set layer2.smceLoss.rows to 6300. this is false but we will exploit our accuracy fn this way
    smceLoss.output.rows = 6300;
    cout<< "TESTset accuracy: " << smceLoss.calc_acc_cpu(smceLoss.output, test_y_true) <<endl;

    // END BLOCK OF CODE TO TEST ACC

    



    

    
    
    CHECK_CUBLAS(cublasDestroy(global_cublas_handle));
    CHECK_CUDNN(cudnnDestroy(global_cudnn_handle));
    return 0;


}

