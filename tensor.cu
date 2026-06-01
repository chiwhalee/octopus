//  tensor.cu (只有这个文件归 NVCC 管，里面可以大方写 CUDA 语法)
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
// 全局静态 cuBLAS 句柄，整个程序生命周期内只初始化一次
static cublasHandle_t global_cublas_handle = nullptr;

//全库唯一的句柄获取器
inline cublasHandle_t get_global_cublas_handle() {
    // 完美的局部静态单例，整个程序生命周期内保证物理上只有一份，且线程安全
    static cublasHandle_t handle = nullptr;
    
    if (handle == nullptr) {
        cublasStatus_t status = cublasCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("🚨 [CUDA 灾难] cuBLAS 句柄创建失败！错误码: %d\n", status);
        } else {
            printf("🚀 [RTX 3090] 全局唯一 cuBLAS 硬件句柄惰性加载成功！\n");
        }
    }
    return handle;
}



// 用纯 C 风格的函数，把 CUDA 的脏活包起来
extern "C" void* allocate_gpu_memory(size_t bytes) {
    void* ptr = nullptr;
    cudaMalloc(&ptr, bytes);
    return ptr;
}

extern "C" void free_gpu_memory(void* ptr) {
    cudaFree(ptr);
}

extern "C" void copy_host_to_device(void* dst, const void* src, size_t bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}

extern "C" {
    void copy_device_to_host(void* dst, const void* src, size_t bytes) {
        // 经典的硬件反向拷贝，依然是同步阻塞的，确保 CPU 拿到数据时一定是热乎的
        cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    }
}


extern "C" {

    // 项目启动或首次使用时调用的初始化
    void init_cublas() {
        if (!global_cublas_handle) {
            cublasCreate(&global_cublas_handle);
        }
    }

    // 项目退出时销毁
    void destroy_cublas() {
        if (global_cublas_handle) {
            cublasDestroy(global_cublas_handle);
            global_cublas_handle = nullptr;
        }
    }

    // ========================================================
    // 针对 double 的 cuBLAS GEMM 纯 C 接口
    // ========================================================
    // 注意：cuBLAS 默认是 列优先（ColMajor） 的，这与你 CPU 侧 Eigen::ColMajor 完美契合！
    void gpu_dgemm_c_interface(
        int m, int n, int k,
        const double* alpha,
        const double* d_A, int lda,
        const double* d_B, int ldb,
        const double* beta,
        double* d_C, int ldc) 
    {
        // 如果句柄忘了初始化，惰性加载一次
        //if (!global_cublas_handle) cublasCreate(&global_cublas_handle);
        cublasHandle_t handle = get_global_cublas_handle();

        // 调用 NVIDIA 官方针对 RTX 3090 Tensor Core 优化的双精度矩阵乘法
        // CUBLAS_OP_N 代表不转置（No Transpose）
        cublasStatus_t status = cublasDgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            alpha,
            d_A, lda,
            d_B, ldb,
            beta,
            d_C, ldc
        );

        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cuBLAS GEMM 报错，错误码: " << status << std::endl;
        }
    }
}


// 限制最大张量阶数为 6，防止在 GPU 上动态分配内存（GPU 核函数内严禁动态分配）
#define MAX_DIMS 6

// ========================================================
// 🔥 【自定义 CUDA Kernel】：硬核多维张量并行置换算子
// ========================================================
__global__ void gpu_permute_kernel(
    const double* src, double* dst, size_t total_elements, int num_dims,
    const int* permute_map,   // 例如 [2, 3, 0, 1]
    const int* src_strides,   // 原张量的步长数组
    const int* dst_dims,      // 目标张量的维度数组
    const int* dst_strides)   // 目标张量的步长数组
{
    // this is for F order,  adjustment needed for C order
    // 1. 算出当前硬件线程的全局绝对索引
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 网格跨步循环，榨干超大张量
    while (idx < total_elements) {
        size_t temp = idx;
        int coords[MAX_DIMS];
        
        
        //2 . ：根据目标张量的一维 idx，反解出它在目标张量里的多维坐标 (k, l, i, j)
        // C order 
        //for (int d = 0; d < num_dims; ++d) {
        //    coords[d] = temp / dst_strides[d];
        //    temp %= dst_strides[d];
        //}
        
        // F order
        for (int d = num_dims - 1; d >= 0; --d) {
            coords[d] = temp / dst_strides[d];
            temp %= dst_strides[d];
        }        
        

        // 3. 【坐标重排】：根据用户传进来的 permute_map，把坐标还原回原张量的顺序 (i, j, k, l)
        int src_coords[MAX_DIMS];
        for (int d = 0; d < num_dims; ++d) {
            src_coords[permute_map[d]] = coords[d];
        }
        
        // 4. 【正向展平】：根据原张量的步长，算出它在原张量显存里的一维绝对地址
        size_t src_idx = 0;
        for (int d = 0; d < num_dims; ++d) {
            src_idx += src_coords[d] * src_strides[d];
        }
        

        // 5. 跨越时空的数据抓取！
        dst[idx] = src[src_idx];

        idx += blockDim.x * gridDim.x;
    }
}

extern "C" {
    // C 桥梁接口
    void gpu_permute_double_c_interface(
        const double* src, double* dst, size_t total_elements, int num_dims,
        const int* permute_map, const int* src_strides, const int* dst_dims, const int* dst_strides) 
    {
        if (total_elements == 0) return;

        // 标准的线程块配置
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        if (blocks > 65535) blocks = 65535;

        // 发射 Kernel！
        gpu_permute_kernel<<<blocks, threads>>>(
            src, dst, total_elements, num_dims,
            permute_map, src_strides, dst_dims, dst_strides
        );
    }
}



// generate 等差数列
__global__ void generate_sequence_kernel(double* ptr, size_t size, double start, double step) {
    // 算出当前网格（Grid）中这个硬件线程的绝对全局全局索引（Global Index）
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 经典的网格跨步循环（Grid-stride loop），确保即使线程数不够，也能把整个超大张量刷完
    while (idx < size) {
        ptr[idx] = start + static_cast<double>(idx) * step;
        idx += blockDim.x * gridDim.x;
    }
}

extern "C" {
    void gpu_custom_sequence_c_interface(void* ptr, size_t size, double start, double step) {
        if (size == 0) return;

        // 设定每个线程块（Block）包含 256 个硬件线程
        int threads_per_block = 256;
        // 算出总共需要多少个线程块，加 255 是为了防止向上取整时漏掉尾部数据
        int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
        
        // 限制最大网格大小，防止开辟过大硬件资源
        if (blocks_per_grid > 65535) blocks_per_grid = 65535;

        // 发射 GPU 算子！
        generate_sequence_kernel<<<blocks_per_grid, threads_per_block>>>(
            static_cast<double*>(ptr), size, start, step
        );
        
        // 提示：由于 CUDA 发射是异步的，如果你需要立刻在 CPU 侧打印它，
        // 可以在这里加上 cudaStreamSynchronize(0); 保证打完收工。
    }
}



void gpu_memset_zero_c_interface(void* ptr, size_t bytes) {
    // 把显存强行刷成 0 
    // 第二个参数 0 代表要把每个字节都刷成 0
    // 在二进制层面，所有位为 0 的 IEEE 754 浮点数就是完美的 0.0
    cudaMemset(ptr, 0, bytes);
}
