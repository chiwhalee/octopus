#pragma once
#include <iostream>
#include <cstddef>
#include <vector> 

// 巧妙的处理：如果不是 NVCC 编译，这两个关键字就是空字符串
// 这样这个头文件就可以被纯 CPU 的 .cpp 文件无缝包含，不会报语法错误
#ifndef __CUDACC__
    #define __host__
    #define __device__
#endif


extern "C" {
    void* allocate_gpu_memory(size_t bytes);
    void  free_gpu_memory(void* ptr);
    void copy_host_to_device(void* dst, const void* src, size_t bytes);
    void copy_device_to_host(void* dst, const void* src, size_t bytes); 
    void gpu_memset_zero_c_interface(void* ptr, size_t bytes); 
    void gpu_dgemm_c_interface(
        int m, int n, int k,
        const double* alpha,
        const double* d_A, int lda,
        const double* d_B, int ldb,
        const double* beta,
        double* d_C, int ldc
    );    
    void gpu_permute_double_c_interface( 
            const double* src, double* dst, 
            size_t total_elements, int num_dims, 
            const int* permute_map, const int* src_strides, 
            const int* dst_dims, const int* dst_strides) ;
    
}



namespace MyGpu {
    template <typename DTYPE>
    class GpuData{
        private:
            DTYPE* _ptr = nullptr;
            size_t _size = 0;
        public:
            GpuData() = default;

            // 显式指定初始大小的构造函数
            explicit GpuData(size_t size) {
                resize(size);
            }

            // 【核心：RAII 析构】确保类生命周期结束时，显存绝对释放
            ~GpuData() {
                if (_ptr) {
                    free_gpu_memory(_ptr);
                }
            }

            // 移动构造函数：允许张量在容器间高效转移（比如 return 或者是 std::move）
            GpuData(GpuData&& other) noexcept : _ptr(other._ptr), _size(other._size) {
                other._ptr = nullptr;
                other._size = 0;
            }

            // 移动赋值运算符
            GpuData& operator=(GpuData&& other) noexcept {
                if (this != &other) {
                    if (_ptr) free_gpu_memory(_ptr);
                    _ptr = other._ptr;
                    _size = other._size;
                    other._ptr = nullptr;
                    other._size = 0;
                }
                return *this;
            }
            
            

            // 【关键安全锁】：禁用普通的拷贝，防止多处隐式拷贝导致 GPU 显存被多次 free
            GpuData(const GpuData&) = delete;
            GpuData& operator=(const GpuData&) = delete;

            // 核心操作方法，模仿 std::vector
            void resize(size_t new_size) {
                if (_ptr) {
                    free_gpu_memory(_ptr);
                    _ptr = nullptr;
                }
                _size = new_size;
                if (_size > 0) {
                    _ptr = static_cast<DTYPE*>(allocate_gpu_memory(_size * sizeof(DTYPE)));
                }
            }

            void fill_zero() {
                    if (_ptr && _size > 0) {
                        // 算好总共需要清空多少个字节（Bytes = 元素个数 * 每个元素占用的字节数）
                        size_t total_bytes = _size * sizeof(DTYPE);
                        
                        gpu_memset_zero_c_interface(static_cast<void*>(_ptr), total_bytes);
                    }
                }            

            void copy_from_host(const std::vector<DTYPE>& host_vec) {
                // 安全第一：防止 CPU vector 大小和当前 GPU 分配的空间对不上
                if (host_vec.size() > _size) {
                    resize(host_vec.size());
                }
                if (!host_vec.empty()) {
                    copy_host_to_device(
                        static_cast<void*>(_ptr), 
                        static_cast<const void*>(host_vec.data()), 
                        host_vec.size() * sizeof(DTYPE)
                    );
                }
            }
            
            // 【形态 A】：传引用，把显存数据灌进一个现有的 CPU vector 里（零新内存开销）
            void copy_to_host(std::vector<DTYPE>& host_vec) const {
                // 安全第一：强行把 CPU 侧的容器 resize 到一模一样大
                host_vec.resize(_size);
                
                if (_size > 0 && _ptr) {
                    copy_device_to_host(
                        static_cast<void*>(host_vec.data()), 
                        static_cast<const void*>(_ptr), 
                        _size * sizeof(DTYPE)
                    );
                }
            }

            // 【形态 B】：传返回值，直接在函数内部临时为你生成一个装满数据的 vector（极度适合单测和打印）
            std::vector<DTYPE> to_vector() const {
                std::vector<DTYPE> host_vec(_size);
                copy_to_host(host_vec); // 复用上面的形态 A
                return host_vec;
            }            
            

            void clear() {
                resize(0);
            }

            // 裸指针与大小访问接口
            DTYPE* data() { return _ptr; }
            const DTYPE* data() const { return _ptr; }  // reload for const type 
            size_t size() const { return _size; }
            bool empty() const { return _size == 0; }
        };

    template <typename DTYPE>
    struct GpuSpan {
        DTYPE* _ptr = nullptr;
        size_t _size = 0;

        // 关键：带上 __host__ __device__ 双工关键字
        // 这样这个 Span 既可以在 Host 端被创建、读取 size，也可以直接作为参数发射进 GPU Kernel
        __host__ __device__ DTYPE* data() const { return _ptr; }
        __host__ __device__ size_t size() const { return _size; }
        __host__ __device__ bool empty() const { return _size == 0; }

        // 在 GPU 内部执行时，允许直接通过 [] 索引显存元素（比如在自定义的 CUDA Kernel 里）
        __host__ __device__ DTYPE& operator[](size_t idx) const {
            return _ptr[idx]; 
        }
        __host__ std::vector<std::remove_const_t<DTYPE>> to_vector() const {
            // 使用 remove_const_t 确保哪怕 T 是 const double，生成的也是 std::vector<double>
            using CPU_TYPE = std::remove_const_t<DTYPE>;
            std::vector<CPU_TYPE> host_vec(_size);
            if (_size > 0 && _ptr) {
                copy_device_to_host(
                    static_cast<void*>(host_vec.data()),
                    static_cast<const void*>(_ptr),
                    _size * sizeof(CPU_TYPE)
                );
            }
            return host_vec;
        }            
        
    };
}

template <typename DTYPE>
std::ostream& operator<<(std::ostream& os, const MyGpu::GpuSpan<DTYPE>& gpu_span) {
    // 可以直接print GpuSpan
    os << "GpuSpan(size=" << gpu_span.size() << ") [";
    // 悄悄拉回 CPU
    auto cpu_data = gpu_span.to_vector();
    for (size_t i = 0; i < cpu_data.size(); ++i) {
        os << cpu_data[i];
        if (i + 1 < cpu_data.size()) os << ", ";
    }
    os << "]";
    return os;
}






namespace not_use_at_the_moment {
    
 template <typename T>
struct DeviceSpan {
    T* ptr;
    size_t len;
    // 带上双工关键字，意味着这个索引操作在 CPU 和 GPU 内部都能跑
    __host__ __device__ T& operator[](size_t i) const {
        return ptr[i]; 
    }
    __host__ __device__ size_t size() const { return len; }
};

   
    
    
    
    
    // 内存管理基础适配器
    template <typename T, typename Device>
    struct DeviceBuffer;
       

    struct CpuDevice {}; 
    struct GpuDevice {}; 



    // CPU 特化版：直接包装 std::vector
    template <typename T>
    struct DeviceBuffer<T, CpuDevice> {
        std::vector<T> storage;

        void resize(size_t size, T init_val = T{}) {
            storage.assign(size, init_val);
        }
        T* data() { return storage.data(); }
        const T* data() const { return storage.data(); }
        size_t size() const { return storage.size(); }
    };

    // GPU 特化版：接管裸指针，实现 RAII
    template <typename T>
    struct DeviceBuffer<T, GpuDevice> {
        T* raw_ptr = nullptr;
        size_t num_elements = 0;

        DeviceBuffer() = default;
        
        // 禁用拷贝，防止双重释放，HPC 容器标准操作
        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;
        
        // 支持移动构造
        DeviceBuffer(DeviceBuffer&& other) noexcept 
            : raw_ptr(other.raw_ptr), num_elements(other.num_elements) {
            other.raw_ptr = nullptr;
            other.num_elements = 0;
        }

        void resize(size_t size, T init_val = T{}) {
            if (raw_ptr) {
                free_gpu_memory(raw_ptr);
                raw_ptr = nullptr;
            }
            num_elements = size;
            if (size > 0) {
                raw_ptr = static_cast<T*>(allocate_gpu_memory(size * sizeof(T)));
                // 注意：如果你的 free_gpu_memory 内部不用 memset 刷0，
                // 且需要初始值为 init_val，这里需要调用一个 GPU kernel 初始化（或从 Host copy）。
                // 假设块初始化在后面算子中处理，或者默认置 0：
            }
        }

        ~DeviceBuffer() {
            if (raw_ptr) {
                free_gpu_memory(raw_ptr);
            }
        }

        T* data() { return raw_ptr; }
        const T* data() const { return raw_ptr; }
        size_t size() const { return num_elements; }
    };
}
