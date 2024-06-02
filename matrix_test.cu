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

    void init_consec() const {
    int count = 0;

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            data[row * cols + col] = count;
            count ++;
        }
    }
}

    // Matrix multiplication
    static void multiply(const Matrix& A, const Matrix& B, Matrix& C) {
    std::cout << "Matrix A: " << A.rows << "x" << A.cols << std::endl;
    std::cout << "Matrix B: " << B.rows << "x" << B.cols << std::endl;
    std::cout << "Matrix C: " << C.rows << "x" << C.cols << std::endl;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // CHECK_CUBLAS(cublasSgemm(global_cublas_handle,
    //                              CUBLAS_OP_N, CUBLAS_OP_N,
    //                              C.cols, C.rows, A.cols, // rows in output, cols in output
    //                              &alpha, // 
    //                              B.data, B.rows, // bdat, ld
    //                              A.data, A.rows, // adat, ld
    //                              &beta,
    //                              C.data, 2)); //cdat, ld

    int m = 2, k = 3, n = 4;


    CHECK_CUBLAS(cublasSgemm(global_cublas_handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 n, m, k, // rows in output, cols in output
                                 &alpha, // 
                                 B.data, n, // bdat, ld
                                 A.data, k, // adat, ld
                                 &beta,
                                 C.data, n)); //cdat, ld






    
}




};

int main(){
    CHECK_CUBLAS(cublasCreate(&global_cublas_handle));
    CHECK_CUDNN(cudnnCreate(&global_cudnn_handle));

    int m = 2, k = 3, n = 4;

    Matrix A(k, m); // A is mxk in row-major
    Matrix B(n, k); // B is kxn in row-major
    Matrix C(n+4, m+4); // C is mxn in row-major (result matrix)

    A.init_consec();
    B.init_consec();
    C.init_zeros();

    std::cout << "Matrix A:" << std::endl;
    A.print();
    std::cout << "Matrix B:" << std::endl;
    B.print();

    Matrix::multiply(A, B, C);

    std::cout << "Matrix C after multiplication:" << std::endl;
    C.print();
    

}