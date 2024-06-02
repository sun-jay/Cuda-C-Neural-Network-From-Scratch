#include <iostream>
#include <cuda_runtime.h>
#include <string>

#include <cublas_v2.h>
#include <cudnn.h>


using namespace std;

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

cublasHandle_t global_cublas_handle;
cudnnHandle_t global_cudnn_handle;

void print(string p){
  cout << p << endl;
}

class CudaTimer {
public:
    CudaTimer(const std::string& name) : name_(name) {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
        cudaEventRecord(start_, 0);
    }

    ~CudaTimer() {
        cudaEventRecord(stop_, 0);
        cudaEventSynchronize(stop_);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_, stop_);
        std::cout << name_ << " - Time elapsed: " << milliseconds / 1000.0 << " seconds" << std::endl;
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

private:
    std::string name_;
    cudaEvent_t start_, stop_;
};

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

__global__ void Softmax_CE_Loss_Kernel(float* output, float* y_true, float* dinputs, int n_inputs, int batch_size) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_tid < batch_size) {
        int idx = global_tid * n_inputs + y_true[global_tid];
        dinputs[idx] = output[idx] - 1.0f;
    }
}

// marker
class Matrix {
public:
    int cols, rows;
    float *data;

    // Constructor
    Matrix(int cols, int rows) : cols(cols), rows(rows) {
        CHECK_CUDA(cudaMallocManaged(&data, cols * rows * sizeof(float)));
    }

    // Destructor
    ~Matrix() {
        cudaFree(data);
    }

    // Access element
    float& operator()(int col, int row) {
        return data[row * cols + col];
    }

    // Access element (const version)
    const float& operator()(int col, int row) const {
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

    // Matrix multiplication
        static void multiply(const Matrix& A, const Matrix& B, Matrix& C, bool transposeA = false, bool transposeB = false) {
                transposeA = !transposeA;
                transposeB = !transposeB;


            cout << "A transpose: "<< transposeA <<endl;
            cout << "B transpose: "<< transposeB <<endl;
            cout << "A:"<<endl;
            A.print();
            cout << "B:"<<endl;
            B.print();
            cout << "C cols " << C.cols << "C rows " << C.rows << endl;


            int colsA = transposeA ? A.rows : A.cols;
            int rowsA = transposeA ? A.cols : A.rows;
            int colsB = transposeB ? B.rows : B.cols;
            int rowsB = transposeB ? B.cols : B.rows;

            // Dimension checks based on transposition
            if (rowsA != colsB) {
                throw std::invalid_argument("Matrix dimensions do not match for multiplication");
            }
            if (colsA != C.cols || rowsB != C.rows) {
                throw std::invalid_argument("Result matrix dimensions do not match for multiplication");
            }

            const float alpha = 1.0f;
            const float beta = 0.0f;

            cublasOperation_t opA = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
            cublasOperation_t opB = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;

            // CHECK_CUBLAS(cublasSgemm(global_cublas_handle,
            //                         opA, opB,
            //                         C.cols, C.rows, rowsA,
            //                         &alpha,
            //                         A.data, A.cols,
            //                         B.data, B.cols,
            //                         &beta,
            //                         C.data, C.cols));
            CHECK_CUBLAS(cublasSgemm(global_cublas_handle,
                                    opA, opB,
                                    C.cols, C.rows, rowsA,
                                    &alpha,
                                    A.data, !transposeA? A.rows : A.cols,
                                    B.data, !transposeA? B.rows : B.cols,
                                    &beta,
                                    C.data, C.rows));
            cudaDeviceSynchronize();
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

    // Constructor
    Layer_Dense(int n_inputs, int n_neurons, int batch_size)
        : weights(n_neurons, n_inputs), biases(n_neurons, 1),
          output(n_neurons, batch_size), dweights(n_neurons, n_inputs),
          dbiases(n_neurons, 1), dinputs(n_inputs, batch_size), inputs(nullptr) {
        weights.init_random();
        biases.init_zeros();
    }

      // Forward pass
      void forward(Matrix& input) {

          // store a pointer to input matrix
          inputs = &input;
          Matrix::multiply(input, weights, output, false, false);

          

          // Add biases -- right now, this isn't accelerated by the GPU. We can add this in later
          for (int i = 0; i < output.cols; i++) {
              for (int j = 0; j < output.rows; j++) {
                  output(i, j) += biases(j, 0);
              }
          }
      }

      // Backward pass
      void backward(const Matrix& dvalues) {
        // cout<<"backward pass"<<endl;
        
        // print("dvalues:");
        // dvalues.print();
        // print("inputs: ");
        // inputs.print();

        // access inputs that we saved in the forward pass

        // std::cout << "dvalues: (" << dvalues.cols << ", " << dvalues.rows << ")" << std::endl;
        // std::cout << "inputs: (" << inputs->cols << ", " << inputs->rows << ")" << std::endl;
        // std::cout << "dweights: (" << dweights.cols << ", " << dweights.rows << ")" << std::endl;
     

        cout<<"GOOD TILL HERE"<<endl;
        Matrix::multiply(*inputs, dvalues, dweights, true, false); // dweights = dvalues * inputs.T
# 
        // cout<<"dweights calculated"<<endl;
        
        // Gradient on biases -- accelerate on GPU later
        for (int i = 0; i < dvalues.cols; i++) {
            for (int j = 0; j < dvalues.rows; j++) {
                dbiases(i, 0) += dvalues(i, j);
            }
        }

        // Gradients on values (inputs)
        Matrix::multiply(dvalues, weights, dinputs, false, true); // dinputs = weights.T * dvalues
    }

    // void column-wise
};

class Activation_Relu{
  public:
  
    Matrix* inputs; 
    Matrix output;

    Matrix dinputs;

    // Constructor
    Activation_Relu(int n_inputs,int batch_size)
        : output(n_inputs, batch_size), dinputs(n_inputs, batch_size), inputs(nullptr)  {

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
    }


};

class Activation_Softmax {
public:
    Matrix* inputs; 
    Matrix output;
    Matrix dinputs;
    cudnnTensorDescriptor_t input_desc, output_desc;

    // Constructor
    Activation_Softmax(int n_inputs, int batch_size)
        : output(n_inputs, batch_size), dinputs(n_inputs, batch_size), inputs(nullptr) {

        CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));

        CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, n_inputs, 1, 1));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, n_inputs, 1, 1));
    }

    // Destructor
    ~Activation_Softmax() {
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc));
    }

    // Forward pass
    void forward(Matrix& input) {
        inputs = &input;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        CHECK_CUDNN(cudnnSoftmaxForward(global_cudnn_handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, input_desc, input.data, &beta, output_desc, output.data));
        cudaDeviceSynchronize();

    }

    // Backward pass
    void backward(const Matrix& dvalues) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        CHECK_CUDNN(cudnnSoftmaxBackward(global_cudnn_handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                                         &alpha, output_desc, output.data, output_desc, dvalues.data,
                                         &beta, input_desc, dinputs.data));
        cudaDeviceSynchronize();
    }


    // Print the output
    void print_output() const {
        output.print();
    }
};

class Softmax_CE_Loss {
public:
    Matrix* inputs; 
    Matrix output;
    Matrix dinputs;
    cudnnTensorDescriptor_t input_desc, output_desc;

    // Constructor
    Softmax_CE_Loss(int n_inputs, int batch_size)
        : output(n_inputs, batch_size), dinputs(n_inputs, batch_size), inputs(nullptr) {

        CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));

        // Configure descriptors for column-major order
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, batch_size, n_inputs, 1 ,1));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT,batch_size,  n_inputs ,1 ,1));
    }

    // Destructor
    ~Softmax_CE_Loss() {
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc));
    }

    // Forward pass
    void forward(Matrix& input) {
        inputs = &input;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        CHECK_CUDNN(cudnnSoftmaxForward(global_cudnn_handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, input_desc, input.data, &beta, output_desc, output.data));

        cudaDeviceSynchronize();

        // we dont actually need to calculate loss, but we should implement this later
    }

    // Backward pass
    void backward(Matrix& dvalues, Matrix& y_true) {

        // y_true will be array of discrete values (index of correct class) with len batch size 
        // we will launch one thread for each sample, because each sample only has 1 element that needs to be updated
        // the thread will index output data at global_tid * len_inputs + y_true[global_tid]
        // and subtract 1 from this value

        int totalLen = y_true.rows;  // Number of samples in the batch
        int blockSize = 1024;
        int numBlocks = (totalLen + blockSize - 1) / blockSize;

        // Launch custom CUDA kernel to compute gradients
        Softmax_CE_Loss_Kernel<<<numBlocks, blockSize>>>(output.data, y_true.data, dinputs.data, output.rows, totalLen);
        cudaDeviceSynchronize();
    }

    // Print the output
    void print_output() const {
        output.print();
    }
};

void checkAvailableMemory() {
    size_t free, total;
    CHECK_CUDA(cudaMemGetInfo(&free, &total));
    std::cout << "GPU Memory - Free: " << free / (1024 * 1024) << " MB, Total: " << total / (1024 * 1024) << " MB" << std::endl;
}

int main() {
    // Initialize cuBLAS handle
    CHECK_CUBLAS(cublasCreate(&global_cublas_handle));
    CHECK_CUDNN(cudnnCreate(&global_cudnn_handle));

    int n_inputs = 3;
    int n_neurons = 2;
    int batch_size = 2;
    

    // make random inputs
    Matrix inputs(n_inputs, batch_size);
    inputs.init_random();
    inputs.print();
    // inputs(0,0) = 1;
    // inputs(1,0) = 2;
    // inputs(2,0) = 3;
    // inputs(0,1) = 4;
    // inputs(1,1) = 5;
    // inputs(2,1) = 6;

    // Random dvalues for backward pass
    Matrix dvalues(n_neurons, batch_size);
    dvalues.init_random();

    Matrix y(1, batch_size );
    for (int i = 0; i<batch_size;i++){
        y(0,i) = 1;
    }
    y.print();
    // y.print();
    
    // // test softmax
    // cout<<"inputs"<<endl;
    // inputs.print();
    // cout<<"inputs rows: "<< inputs.rows<<endl;

    
    // Softmax_CE_Loss softmax_ceLoss(n_inputs, batch_size);
    // {CudaTimer("fwd bkwd softmax");

    //     softmax_ceLoss.forward(inputs);


    //     // softmax_ceLoss.backward(dvalues, y);
    // }

    //     softmax_ceLoss.output.print();

    // return 0;
    

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Record the start event
    CHECK_CUDA(cudaEventRecord(start));

    // Initialize dense layer
    Layer_Dense layer(n_inputs, n_neurons, batch_size);

    // Forward pass
    layer.forward(inputs);




    // Initialize activation function
    Activation_Relu activation(n_neurons, batch_size);

    // Apply activation function
    activation.forward(layer.output);

    // Backward pass through activation function
    activation.backward(dvalues);

    // Backward pass through dense layer
    layer.backward(activation.dinputs);

    return 0;



    // Record the stop event
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // Print the elapsed time
    std::cout << "Time elapsed: " << milliseconds / 1000.0f  << " seconds" << std::endl;

    // Clean up CUDA events
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    // Clean up cuBLAS handle
    CHECK_CUBLAS(cublasDestroy(global_cublas_handle));

    return 0;
}
