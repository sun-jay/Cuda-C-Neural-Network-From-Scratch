#ifndef NN_H
#define NN_H

#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include <chrono>

#include <cublas_v2.h>
#include <cudnn.h>


using namespace std;

cublasHandle_t global_cublas_handle;
cudnnHandle_t global_cudnn_handle;

#define CHECK_LAST_CUDA_ERROR() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA kernel launch error at " << __FILE__ << ":" << __LINE__ << " code=" << err << " (" << cudaGetErrorString(err) << ")" << std::endl; \
        exit(1); \
    } \
}
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__; \
        std::cerr << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t err = call; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__; \
        std::cerr << " code=" << err << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_CUDNN(call)                                          {                                                                             \
    cudnnStatus_t err = call;                                                 \
    if (err != CUDNN_STATUS_SUCCESS) {                                        \
        std::cerr << "cuDNN error in file '" << __FILE__ << "' at line " << __LINE__ \
                  << " : " << cudnnGetErrorString(err) << std::endl;          \
        std::exit(EXIT_FAILURE);                                              \
    }                                                                         \
}

void print(string p) {
    cout << p << endl;
}

// relu forward kernel
static __global__ void relu_forward(float *inputs, float *outputs, int size){
  int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_tid<size)
    outputs[global_tid] = fmaxf(0.0f,inputs[global_tid]);

}

// relu backward kernel
static __global__ void relu_backward(float *dvalues, float *dinputs, float *inputs, int size){
  int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_tid < size) {
        dinputs[global_tid] = (inputs[global_tid] > 0) ? dvalues[global_tid] : 0.0f;
    }
}

// Softmax_CE_Loss backward kernel
__global__ void Softmax_CE_Loss_Kernel(float* output, float* y_true, float* dinputs, int n_inputs, int batch_size) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    // batch size is the len of y_true

    if (global_tid < batch_size) {
        // idx will refer to the element that should have been true

        int idx = global_tid * n_inputs + static_cast<int>(y_true[global_tid]);
        dinputs[idx] = output[idx] - 1.0f;
    }
}

__global__ void update_weights_kernel(float* weights, float* weight_momentums, float* weight_cache,
        float* dweights, int totalLen, float beta_1, float beta_2,
        float current_learning_rate, float epsilon, int iterations){

    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_tid < totalLen) {

        weight_momentums[global_tid] = beta_1 * weight_momentums[global_tid] + (1 - beta_1) * dweights[global_tid];

        // write this out in comments for readability but make code succint for efficiency -- we will need this definition to calc weights adjustment

        // weight_momentums_corrected[global_tid] = weight_momentums[global_tid] / (1 - beta_1 ** (iterations + 1))
        // weight_cache_corrected[global_tid] = weight_cache[global_tid] / (1 - beta_2 ** (iterations + 1))

        weight_cache[global_tid] = beta_2 * weight_cache[global_tid] + (1 - beta_2) * pow(dweights[global_tid], 2);


        // layer.weights += -self.current_learning_rate * \
        //             weight_momentums_corrected / \
        //             (np.sqrt(weight_cache_corrected) +
        //                 self.epsilon)
        float del = -current_learning_rate
        * (weight_momentums[global_tid] / (1 -   pow (beta_1,(iterations + 1))     ))
        / (pow(      weight_cache[global_tid] / (1 - pow(beta_2, (iterations + 1)))   ,0.5    ) + epsilon);

        // printf("tid: %d, delta: %f\n", global_tid, del);

        weights[global_tid] += -current_learning_rate
        * (weight_momentums[global_tid] / (1 -   pow (beta_1,(iterations + 1))     ))
        / (pow(      weight_cache[global_tid] / (1 - pow(beta_2, (iterations + 1))      ),0.5    ) + epsilon);


    }
}
// __global__ void update_biases_kernel(float* biases, float* bias_momentums, float* bias_cache,
//         float* dbiases, int totalLen, float beta_1, float beta_2,
//         float current_learning_rate, float epsilon, int iterations){

//     int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (global_tid < totalLen) {

//         bias_momentums[global_tid] = beta_1 * biases_momentums[global_tid] + (1 - beta_1) * dbiases[global_tid];

//         // write this out in comments for readability but make code succint for efficiency -- we will need this definition to calc biases adjustment

//         // bias_momentums_corrected[global_tid] = bias_momentums[global_tid] / (1 - beta_1 ** (iterations + 1))
//         // bias_cache_corrected[global_tid] = bias_cache[global_tid] / (1 - beta_2 ** (iterations + 1))

//         bias_cache[global_tid] = beta_2 * bias_cache[global_tid] + (1 - beta_2) * pow(dbiases[global_tid], 2);


//         // layer.biases += -self.current_learning_rate * \
//         //             bias_momentums_corrected / \
//         //             (np.sqrt(bias_cache_corrected) +
//         //                 self.epsilon)
//         float del = -current_learning_rate
//         * (bias_momentums[global_tid] / (1 -   pow (beta_1,(iterations + 1))     ))
//         / (pow(      bias_cache[global_tid] / (1 - pow(beta_2, (iterations + 1)))   ,0.5    ) + epsilon);

//         // printf("tid: %d, delta: %f\n", global_tid, del);

//         biases[global_tid] += -current_learning_rate
//         * (bias_momentums[global_tid] / (1 -   pow (beta_1,(iterations + 1))     ))
//         / (pow(      bias_cache[global_tid] / (1 - pow(beta_2, (iterations + 1))      ),0.5    ) + epsilon);


//     }
// }

class CudaTimer {
public:
    CudaTimer(const std::string& name, float& total_time_ref, int e = -1, int mN = 10) 
        : name_(name), total_time_ref_(&total_time_ref), epoch(e), modNum(mN) {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
        cudaEventRecord(start_, 0);
    }

    CudaTimer(const std::string& name) 
        : name_(name), total_time_ref_(nullptr), epoch(-1), modNum(10) {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
        cudaEventRecord(start_, 0);
    }

    ~CudaTimer() {
        cudaEventRecord(stop_, 0);
        cudaEventSynchronize(stop_);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_, stop_);
        float seconds = milliseconds / 1000.0;
        if (epoch != -1 && epoch % modNum == 0) {
            std::cout << name_ << " Epoch: " << epoch << " - Time elapsed: " << seconds << " seconds" << std::endl;
        }
        if (total_time_ref_) {
            *total_time_ref_ += seconds;
        }
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

private:
    std::string name_;
    cudaEvent_t start_, stop_;
    float* total_time_ref_;
    int epoch;
    int modNum;
};

class StdTimer {
public:
    StdTimer(const std::string& name, float& total_time_ref, int e = -1, int mN = 10) 
        : name_(name), total_time_ref_(&total_time_ref), epoch(e), modNum(mN) {
        start_ = std::chrono::high_resolution_clock::now();
    }

    StdTimer(const std::string& name) 
        : name_(name), total_time_ref_(nullptr), epoch(-1), modNum(10) {
        start_ = std::chrono::high_resolution_clock::now();
    }

    ~StdTimer() {
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = stop - start_;
        float seconds = duration.count();
        if (epoch != -1 && epoch % modNum == 0) {
            std::cout << name_ << " Epoch: " << epoch << " - Time elapsed: " << seconds << " seconds" << std::endl;
        }
        if (total_time_ref_) {
            *total_time_ref_ += seconds;
        }
    }

private:
    std::string name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    float* total_time_ref_;
    int epoch;
    int modNum;
};


class Matrix {
public:
    int rows, cols;
    float *data;
    bool cpu_only;

    // Constructor
    Matrix(int rows, int cols, bool cpu_only = false) 
        : rows(rows), cols(cols), cpu_only(cpu_only) {
        if (cpu_only) {
            data = new float[rows * cols];
        } else {
            CHECK_CUDA(cudaMallocManaged(&data, rows * cols * sizeof(float)));
        }
    }

    // Destructor
    ~Matrix() {
        if (cpu_only) {
            delete[] data;
        } else {
            cudaFree(data);
        }
    }

    // Access element
    float& operator()(int row, int col) {
        return data[row * cols + col];
    }

    // Access element (const version)
    const float& operator()(int row, int col) const {
        return data[row * cols + col];
    }

    // Print matrix
    void print(int limit = INT_MAX) const {
        int count = 0;

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            std::cout << data[row * cols + col] << " ";
            count ++;
            if (count > limit){
                return;
            }
        }
        std::cout << std::endl;
    }
    }

    void printDims(string name)  const {
        cout << name << " Rows: " << rows << "Cols: " << cols <<endl;
    }



    // Initialize matrix with zeros
    void init_zeros() {
        for (int i = 0; i < cols * rows; ++i) {
            data[i] = 0.0f;
        }
    }

    // Initialize matrix with random values scaled by 0.01
    void init_random() {
        for (int i = 0; i < cols * rows; ++i) {
            data[i] = 10.0f * static_cast<float>(rand()) / RAND_MAX;
        }
    }

    void init_consec() const {
        int count = 0;

        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                data[row * cols + col] = count;
                count ++;
            }
        }
    }

    void init_int(int val){
        for (int i = 0; i < rows*cols; i++){
            data[i] = val;
        }
    }

    // Matrix multiplication
    static void multiply(const Matrix& A, const Matrix& B, Matrix& C, bool transpose_A = false, bool transpose_B = false) {

    if (A.cpu_only || B.cpu_only){
        cout << "CPU ONLY MATRIX!!";
    }

    int Rows_A = transpose_A ? A.cols : A.rows;
    int Cols_A = transpose_A ? A.rows : A.cols;
    int Rows_B = transpose_B ? B.cols : B.rows;
    int Cols_B = transpose_B ? B.rows : B.cols;

    if (Cols_A != Rows_B){
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }

    if (C.rows != Rows_A || C.cols != Cols_B){
        throw std::invalid_argument("C dimensions are not compatible with A and B");
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // this will handle row major and carry out the mm like normal
    CHECK_CUBLAS(cublasSgemm(global_cublas_handle,
            transpose_B ? CUBLAS_OP_T : CUBLAS_OP_N ,
            transpose_A ? CUBLAS_OP_T : CUBLAS_OP_N ,
            C.cols , C.rows, transpose_A ? A.rows : A.cols,
            &alpha,
            B.data, B.cols,
            A.data, A.cols,
            &beta,
            C.data,C.cols));

    // **** SUPER IMPORTANT        
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    }

};

class Layer_Dense {
  public:
    Matrix weights;
    Matrix biases;
    Matrix output;
    Matrix dweights;
    Matrix dbiases;
    Matrix dinputs;
    Matrix* inputs; 

    // for optimizer
    Matrix weight_momentums;
    Matrix weight_cache;
    Matrix bias_momentums;
    Matrix bias_cache;




    // Constructor
    Layer_Dense(int batch_size, int n_inputs, int n_neurons):
        weights(n_inputs, n_neurons),
        biases(1, n_neurons),
        output(batch_size, n_neurons),
        dweights(n_inputs, n_neurons),
        dbiases(1, n_neurons), 
        dinputs(batch_size, n_inputs), 
        inputs(nullptr),
        
        // for optimizer
        weight_momentums(n_inputs, n_neurons),
        weight_cache(n_inputs, n_neurons),
        bias_momentums(1, n_neurons),
        bias_cache(1, n_neurons)

        {

        weights.init_random();
        biases.init_consec();
    }

      // Forward pass
    void forward(Matrix& input) {

        // store a pointer to input matrix
        inputs = &input;

        Matrix::multiply(input, weights, output, false, false);

        // Add biases -- right now, this isn't accelerated by the GPU. We can add this in later
        for(int col = 0; col<output.cols; col++){
            for(int row = 0; row<output.rows; row++){
                output(row,col) += biases(0,col);
            }
        }
        
    }

      // Backward pass
    void backward(const Matrix& dvalues) {

        // inputs->printDims("inpts");
        // dvalues.printDims("dvalues");
        // dweights.printDims("dweights");

        Matrix::multiply( *inputs, dvalues, dweights, true, false);
        Matrix::multiply( dvalues, weights, dinputs, false, true);
        
        // Add biases -- right now, this isn't accelerated by the GPU. We can add this in later
        for(int col = 0; col<output.cols; col++){
            for(int row = 0; row<output.rows; row++){
                dbiases(0,col) += dvalues(row,col);
            }
        }

    }

};

class Activation_Relu{
  public:
  
    Matrix* inputs; 
    Matrix output;

    Matrix dinputs;

    // Constructor
    Activation_Relu(int batch_size, int n_inputs)
        : output(batch_size, n_inputs), dinputs(batch_size, n_inputs), inputs(nullptr)  {

    }

    void forward(Matrix& input){
      inputs = &input;
      
      // run kernel to load outputs with inputs all set to 0
      // kernel(input, outputs, input.cols*input.rows)

      int totalLen = input.cols*input.rows;

      int blockSize = 1024;

      int numBlocks = (totalLen + blockSize - 1)/blockSize;

      relu_forward <<< numBlocks, blockSize >>> (input.data, output.data, totalLen);

      cudaDeviceSynchronize();
      CHECK_LAST_CUDA_ERROR();

    }

    void backward(Matrix& dvalues){

      // dvalues will be of (n_inputs, batch_size), same as everything else for relu because it is element wise
      // run kernel to load dinputs with 0s where dvalues is negative

      // kernel(dvalues,dinputs, input.cols*input.rows)

      int totalLen = dvalues.cols*dvalues.rows;

      int blockSize = 1024;

      int numBlocks = (totalLen + blockSize - 1)/blockSize;

      relu_backward <<< numBlocks, blockSize >>> (dvalues.data, dinputs.data, inputs->data, totalLen);

      cudaDeviceSynchronize();
      CHECK_LAST_CUDA_ERROR();
    }


};

class Softmax_CE_Loss {
public:
    Matrix* inputs; 
    Matrix output;
    Matrix dinputs;
    cudnnTensorDescriptor_t input_desc, output_desc;

    // Constructor
    Softmax_CE_Loss(int batch_size, int n_inputs)
        : output(batch_size, n_inputs), dinputs(batch_size, n_inputs), inputs(nullptr) {

        CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));

        // Configure descriptors for column-major order
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,  batch_size, n_inputs, 1, 1)); 
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, n_inputs, 1, 1));

    }

    // Destructor
    ~Softmax_CE_Loss() {
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc));
    }

    // Forward pass
    void forward(Matrix& input) {
        inputs = &input;
        float alpha = 1.0f;
        float beta = 0.0f;
        // input.printDims("inp");
        // output.printDims("oyt");
        CHECK_CUDNN(cudnnSoftmaxForward(global_cudnn_handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, input_desc, input.data, &beta, output_desc, output.data));

        cudaDeviceSynchronize();
        CHECK_LAST_CUDA_ERROR();

        // we dont actually need to calculate loss, but we should implement this later
    }

    float calc_loss_cpu(Matrix& y_pred, Matrix& y_true){
        { // scoped block so matrices are destroyed

        Matrix y_pred_clipped(y_pred.rows, y_pred.cols, true);

        for (int i = 0; i < y_pred.rows; ++i) {
            for (int j = 0; j < y_pred.cols; ++j) {
                y_pred_clipped(i, j) = std::min(std::max(y_pred(i, j), 1e-7f), 1.0f - 1e-7f);
        
            }
        }

        Matrix correct_confidences(1,y_true.cols, true);

        for (int i = 0; i < y_true.cols; ++i) {
            int true_class = static_cast<int>(y_true(0, i));
            correct_confidences(0,i) = y_pred_clipped(i, true_class);
        }

        float sum;

        for (int i = 0; i < y_true.cols; ++i) {
            sum += -std::log(correct_confidences(0,i));
        }
        return sum/y_true.cols;
        }

    }

    float calc_acc_cpu(const Matrix& y_pred, const Matrix& y_true) {
        float total_correct = 0;

        // For each sample prediction
        for (int i = 0; i < y_pred.rows; ++i) {
            int true_class_index = static_cast<int>(y_true(0, i));

            total_correct ++;

            // Compare to each class prediction in the sample
            for (int j = 0; j < y_pred.cols; ++j) {
                if (y_pred(i, true_class_index) < y_pred(i, j)) {
                    total_correct--;
                    break;
                }
            }

        }

        // Finally divide by total samples (y_pred.rows)
        return total_correct / y_pred.rows;
    }

    // Backward pass
    void backward(Matrix& dvalues, Matrix& y_true) {

        // y_true will be array of discrete values (index of correct class) with len batch size 
        // we will launch one thread for each sample, because each sample only has 1 element that needs to be updated
        // the thread will index output data at global_tid * len_inputs + y_true[global_tid]
        // and subtract 1 from this value

        int totalLen = y_true.cols;  // Batch size
        int blockSize = 1024;
        int numBlocks = (totalLen + blockSize - 1) / blockSize;

        // Launch custom CUDA kernel to compute gradients
        Softmax_CE_Loss_Kernel<<<numBlocks, blockSize>>>(output.data, y_true.data, dinputs.data, output.cols, totalLen);
        cudaDeviceSynchronize();
        CHECK_LAST_CUDA_ERROR();

    }


};

class Optimizer_Adam {
public:
    float learning_rate;
    float current_learning_rate;
    float decay;
    int iterations;
    float epsilon;
    float beta_1;
    float beta_2;

    // Constructor
    Optimizer_Adam(float learning_rate = 0.001, float decay = 0.0, float epsilon = 1e-7,
                   float beta_1 = 0.9, float beta_2 = 0.999)
        : learning_rate(learning_rate), current_learning_rate(learning_rate),
          decay(decay), iterations(0), epsilon(epsilon), beta_1(beta_1), beta_2(beta_2) {}

    void pre_update_params() {
        if (decay) {
            current_learning_rate = learning_rate * (1.0 / (1.0 + decay * iterations));
        }
    }

    void update_params(Layer_Dense& layer) {
        // total number of elements in w and b
        int weight_size = layer.weights.rows * layer.weights.cols;
        int biases_size = layer.biases.rows * layer.biases.cols;

        // Define the number of threads per block and blocks per grid for the GPU kernels
        int threads_per_block = 1024;
        // this will assign one thread to each w and b (dont for get to add inbound check in the kernel)
        int blocks_per_grid_weights = (weight_size + threads_per_block - 1) / threads_per_block;
        int blocks_per_grid_biases = (biases_size + threads_per_block - 1) / threads_per_block;

        // Launch kernel to update weights, weight momentums, and weight cache

        // cout<<"weightsize" << weight_size <<endl;

        update_weights_kernel<<<blocks_per_grid_weights, threads_per_block>>>(layer.weights.data, layer.weight_momentums.data, layer.weight_cache.data,
        layer.dweights.data, weight_size, beta_1, beta_2,
        current_learning_rate, epsilon, iterations);


        cudaDeviceSynchronize();
        CHECK_LAST_CUDA_ERROR();

        update_weights_kernel<<<blocks_per_grid_biases, threads_per_block>>>(layer.biases.data, layer.bias_momentums.data, layer.bias_cache.data,
        layer.dbiases.data, biases_size, beta_1, beta_2,
        current_learning_rate, epsilon, iterations);

        cudaDeviceSynchronize();
        CHECK_LAST_CUDA_ERROR();

    }

    void post_update_params() {
        iterations++;
    }
};

#endif

// int main(){
//     CHECK_CUBLAS(cublasCreate(&global_cublas_handle));
//     CHECK_CUDNN(cudnnCreate(&global_cudnn_handle));

//     // hts(n_neurons, n_inputs), biases(n_neurons, 1),
//     //       output(n_neurons, batch_size), dweights
//     int batch_size = 30;
//     int n_inputs = 3;
//     int n_neurons = 5;

//     // rows, cols

//     Matrix inputs(batch_size, n_inputs);
//     inputs.init_random();

//     Matrix y_true(1, batch_size);
//     y_true.init_int(1);

    
//     Layer_Dense layer1(batch_size, n_inputs, n_neurons);
//     Activation_Relu act1(batch_size, n_neurons);
//     Softmax_CE_Loss smceLoss(batch_size, n_neurons);
//     Optimizer_Adam optimizer(0.05, 5e-7);
    

//     for(int epoch = 0;  epoch <100; epoch ++){
    
//     {CudaTimer("fwd bkwd");
    
//     layer1.forward(inputs);
//     act1.forward(layer1.output);
//     smceLoss.forward(act1.output);
    
//     smceLoss.backward(smceLoss.output, y_true);
//     act1.backward(smceLoss.dinputs);
//     layer1.backward(act1.dinputs);

//     optimizer.pre_update_params();
//     optimizer.update_params(layer1);
//     optimizer.post_update_params();

//     cout<<"epoch: " << epoch << "loss: " << smceLoss.calc_loss_cpu(smceLoss.output, y_true) <<endl;

//     }

//     }
//     // dvalues.print();
//     // smceLoss.output.print();
//     // layer1.dinputs.print();
//     // cout<<"loss: " << loss <<endl;
//     // layer1.dinputs.printDims("layer1 dinp");


    
// }




