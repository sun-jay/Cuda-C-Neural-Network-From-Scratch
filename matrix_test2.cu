#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <climits>

using namespace std;
cublasHandle_t global_cublas_handle;


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

class Matrix {
public:
    int rows, cols;
    float *data;

    // Constructor
    Matrix(int rows, int cols) : rows(rows),cols(cols) {
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
    static void multiply(const Matrix& A, const Matrix& B, Matrix& C, bool transpose_A = false, bool transpose_B = false) {

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
    cublasSgemm(global_cublas_handle,
            transpose_B ? CUBLAS_OP_T : CUBLAS_OP_N ,
            transpose_A ? CUBLAS_OP_T : CUBLAS_OP_N ,
            C.cols , C.rows, transpose_A ? A.rows : A.cols,
            &alpha,
            B.data, B.cols,
            A.data, A.cols,
            &beta,
            C.data,C.cols);

    // **** SUPER IMPORTANT        
    cudaDeviceSynchronize();

    }

};


int main() {

    CHECK_CUBLAS(cublasCreate(&global_cublas_handle));
    
    int m = 2; // rows in A and rows in C
    int k = 4; // cols in A and rows in B
    int n = 3; // cols in B and cols in C

    Matrix A(m,k);
    Matrix B(k,n);
    Matrix C(m,n);

    A.init_consec();
    B.init_consec();

    cout << "A:"<< endl;
    A.print();
    cout << "B:"<< endl;
    B.print();

    Matrix::multiply(A,B,C, false, false);

    cout << "C:"<< endl;
    C.print();

    CHECK_CUBLAS(cublasDestroy(global_cublas_handle)); // Destroy cuBLAS context

    return 0;
}

// int main() {
//     int m = 2; // rows in A
//     int k = 4; // cols in A and rows in B
//     int n = 3; // rows in B
//     int print = 1;
//     cublasHandle_t handle;

//     float *a, *b, *c;

//     // Allocate memory for a, b, c
//     a = (float*)malloc(m * k * sizeof(float));
//     b = (float*)malloc(k * n * sizeof(float));
//     c = (float*)malloc(m * n * sizeof(float));

//     // Initialize matrix a row by row
//     int ind = 0;
//     for(int j = 0; j < m * k; j++) {
//         a[j] = (float)ind++;
//     }

//     // Initialize matrix b column by column
//     ind = 0;
//     for(int j = 0; j < k * n; j++) {
//         b[j] = (float)ind++;
//     }

//     // Device memory
//     float *d_a, *d_b, *d_c;
//     CHECK_CUDA(cudaMalloc((void**)&d_a, m * k * sizeof(float)));
//     CHECK_CUDA(cudaMalloc((void**)&d_b, k * n * sizeof(float)));
//     CHECK_CUDA(cudaMalloc((void**)&d_c, m * n * sizeof(float)));

//     CHECK_CUBLAS(cublasCreate(&handle)); // Initialize cuBLAS context

//     // Copy matrices a and b to device
//     CHECK_CUBLAS(cublasSetMatrix(m, k, sizeof(*a), a, m, d_a, m));
//     CHECK_CUBLAS(cublasSetMatrix(k, n, sizeof(*b), b, k, d_b, k));

//     float al = 1.0f;
//     float bet = 0.0f;

//     // Perform matrix multiplication: C = al * A * B + bet * C
//     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,m,k,&al,d_b,n,d_a,k,&bet,d_c,n);

//     // Copy result matrix d_c back to host c
//     CHECK_CUBLAS(cublasGetMatrix(m, n, sizeof(*c), d_c, m, c, m));

//     // Print the result matrix
//     if(print == 1) {
//         printf("\nc after Sgemm:\n");
//         for(int i = 0; i < m * n; i++) {
//             printf("%f ", c[i]);
//         }
//         printf("\n");
//     }

//     // Free device memory
//     CHECK_CUDA(cudaFree(d_a));
//     CHECK_CUDA(cudaFree(d_b));
//     CHECK_CUDA(cudaFree(d_c));

//     // Free host memory
//     free(a);
//     free(b);
//     free(c);

//     CHECK_CUBLAS(cublasDestroy(handle)); // Destroy cuBLAS context

//     return 0;
// }





    // cublasSgemm(global_cublas_handle,
    //         transpose_A ? CUBLAS_OP_T : CUBLAS_OP_N ,
    //         transpose_B ? CUBLAS_OP_T : CUBLAS_OP_N ,
    //         transpose_B ? B.rows : B.cols, transpose_A ? A.cols : A.rows, transpose_A ? A.rows : A.cols,
    //         &alpha,
    //         B.data, B.cols,
    //         A.data, A.cols,
    //         &beta,
    //         C.data,transpose_B ? B.rows : B.cols);


    // cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,m,k,&al,d_b,n,d_a,k,&bet,d_c,n)


    //  cublasSgemm(global_cublas_handle,
    //         transpose_A ? CUBLAS_OP_T : CUBLAS_OP_N ,
    //         transpose_B ? CUBLAS_OP_T : CUBLAS_OP_N ,
    //         B.cols,A.rows,A.cols,
    //         &alpha,
    //         B.data,B.cols,
    //         A.data,A.cols,
    //         &beta,
    //         C.data,B.cols);

