#pragma once  // hpp file only compile once 

//#include <Eigen/src/Core/Map.h>
//#include <Eigen/src/Core/Matrix.h>
//#include <Eigen/src/Core/util/Constants.h>
#include <cstdio>
#include <format>
#include <iostream>
//#include <ratio>
#include <string>
#include <vector>
#include <algorithm>   // reverse,  etc
#include <variant>  // std::variant 
#include <span>  //std::span  // chunck of mem window 
#include <Eigen/Dense>                 
#include <Eigen/Core>

#include <cuda_runtime.h>
#include <cublas_v2.h>


#include <cassert>

#include <concepts>
#include <type_traits>

#include "utilities.cpp"
#include "utilities.hpp"
#include "quantum_number.hpp"
#include "gpu_helper.hpp"


extern "C" void* allocate_gpu_memory(size_t bytes);
extern "C" void free_gpu_memory(void* ptr);
extern "C" void copy_host_to_device(void* dst, const void* src, size_t bytes);



inline void ndindex(const std::vector<int>& shape) {
    int rank = shape.size();
    std::vector<int> current_idx(rank);
    bool stop = false;
    
    while (!stop) {
        // 输出当前的坐标 (相当于 np.ndindex 的输出)
        
        //for(int i : current_idx) std::cout << i << " ";
        //std::cout << std::endl;

        // 模拟进位逻辑 (从最后一维开始加 1)
        for (int i = rank - 1; i >= 0; --i) {
            current_idx[i]++;
            if (current_idx[i] < shape[i]) {
                stop = false;
                break;
            } 
            else {
                current_idx[i] = 0;
                if (i == 0) stop = true; // 遍历结束
            }
        }
    }
}


inline std::vector<int> ndindex_F_order(const std::vector<int>& shape) {
    int rank = shape.size();
    std::vector<int> current_idx(rank);
    bool stop = false;
    
    size_t tot_dim = 1;
    for (int dim : shape) tot_dim *= dim;    
    
    //std::cout << tot_dim;
    
    //KEEP THIS: because of 'RVO' (return value optimization), not need to return a pointer
    //std::vector<std::vector<int>> res(tot_dim);
    std::vector<int> res(tot_dim*rank);
    int aa=0;
    int* ptr= res.data(); 
    while (!stop) {
        
        // 输出当前的坐标 (相当于 np.ndindex 的输出)
        //for(int i : current_idx) std::cout << i << " ";
        //std::cout << std::endl;
        //std::span<int> qn_ind_tuple = std::span(res).subspan(aa*rank, rank);
        for(int k=0; k<rank; k++){
            ptr[k] = current_idx[k];
        }
        ptr += rank; 
        aa++;
        for (int i = 0; i <rank; i++) {
            current_idx[i]++;
            if (current_idx[i] < shape[i]) {
                stop = false;
                break;
            } 
            else {
                current_idx[i] = 0;
                if (i == rank-1) {
                    stop = true; // 遍历结束
                }
            }
        }
    }
    return res; 
}


inline std::vector<int> calc_stride_F_order(const std::vector<int> & shape) {
    auto rank=shape.size();
    std::vector<int> stride;
    stride.push_back(1);
    for(int i=1; i<rank; ++i){
        //stride *= dims[i]; 
        stride.push_back(stride[i-1]*shape[i-1]);
    }
    return stride ;
}

inline std::vector<int> calc_stride_C_order(const std::vector<int>& d) {
    std::vector<int> s(d.size(), 1);
    for (int i = d.size() - 2; i >= 0; --i) {
        s[i] = s[i + 1] * d[i + 1];
    }
    return s; 
}



// equivalent to np.ravel_multi_index for 'F' order
inline int ravel_multi_index_F_order(const std::vector<int>& indices, const std::vector<int>& dims) {
    int flat_index = 0;
    int stride = 1;
    for (int i = 0; i < indices.size(); ++i) {
        flat_index += indices[i] * stride;
        stride *= dims[i]; 
    }
    return flat_index;
}



//template <typename DTYPE> 
//void transpose_dense(std::span<const DTYPE> array_flat, const std::vector<int>& shape,  
//                const std::vector<int>& order,  std::span<DTYPE> out) {
//    int rank = shape.size();
//    std::vector<int> new_shape(rank);
//    for(int i=0; i<rank; ++i) new_shape[i] = shape[order[i]];
//
//    // 简单的多维遍历实现（HPC 中可进一步使用递归或 Tiling 优化）
//    std::vector<int> it_idx(rank, 0);
//    int total_size = array_flat.size();
//    
//    for (int i = 0; i < total_size; ++i) {
//        // 1. 计算当前 i 在 array_flat 中的多维索引 (Fortran Order)
//        // 2. 根据 order 映射到 out 的多维索引
//        // 3. 计算 out 中的平铺索引并赋值
//        
//        // 这里为了清晰采用最直观的逻辑：
//        std::vector<int> out_idx(rank);
//        for(int r=0; r<rank; ++r) out_idx[r] = it_idx[order[r]];
//        
//        int array_flat_flat = ravel_multi_index_F_order(it_idx, shape);
//        int out_flat = ravel_multi_index_F_order(out_idx, new_shape);
//        
//        out[out_flat] = array_flat[array_flat_flat];
//
//        // 更新 Fortran order 迭代器
//        for (int r = 0; r < rank; ++r) {
//            if (++w[r] < shape[r]) break;
//            it_idx[r] = 0;
//        }
//    }
//}


template <typename DTYPE> 
void transpose_dense(std::span<const DTYPE> array_flat, const std::vector<int>& shape,  
                    const std::vector<int>& order, std::span<DTYPE> out) {
    int rank = shape.size();
    std::vector<int> new_shape(rank);
    for(int i=0; i<rank; ++i) new_shape[i] = shape[order[i]];

    std::vector<int> in_idx(rank, 0);
    std::vector<int> out_idx(rank); // 在循环外分配空间
    int total_size = array_flat.size();
    auto stride = calc_stride_F_order(new_shape);
    //print_vec(stride, "stride");
    
    for (int i = 0; i < total_size; ++i) {
        
        int i_out=0;
        // 映射索引
        for(int r=0; r<rank; ++r) {
            out_idx[r] = in_idx[order[r]];
            i_out += out_idx[r] * stride[r];
        }
        
        //int i_out = ravel_multi_index_F_order(out_idx, new_shape);
        out[i_out] = array_flat[i];

        // 更新 Fortran order 迭代器
        for (int r = 0; r < rank; ++r) {
            if (++in_idx[r] < shape[r]) break;
            in_idx[r] = 0;
        }
    }
}



template <typename DTYPE> 
void transpose_dense_gpu(
    MyGpu::GpuSpan<const DTYPE> data_flat, 
    const std::vector<int>& shape, 
    std::vector<int> h_order, 
    MyGpu::GpuSpan<DTYPE> out){
    // 1. 在 CPU 侧计算好必要的元数据（维度和步长，因为数据量极小，一般就4个数字，CPU算完传过去

    int rank = shape.size();
    std::vector<int> new_shape(rank);
    for(int i=0; i<rank; ++i) new_shape[i] = shape[h_order[i]];  // 新张量的维度
   
    std::vector<int> h_src_strides = calc_stride_F_order(shape);  // 根据原维度算出的步长
    std::vector<int> h_dst_strides = calc_stride_F_order(new_shape);  // 新张量的步长

    // 2. 利用 Host-Side Staging，把这几个小元数据通过你的内存泵送进 GPU
    // 为了极致性能，这几个元数据可以临时申请极小的 GPU 内存，或者做成类的成员
    MyGpu::GpuData<int> d_order(h_order.size());
    MyGpu::GpuData<int> d_src_strides(h_src_strides.size());
    MyGpu::GpuData<int> d_dst_dims(new_shape.size());
    MyGpu::GpuData<int> d_dst_strides(h_dst_strides.size());
    
    d_order.copy_from_host(h_order);
    d_src_strides.copy_from_host(h_src_strides);
    d_dst_dims.copy_from_host(new_shape);
    d_dst_strides.copy_from_host(h_dst_strides);

    // 3. 准备新张量 t3 的 GPU 空间（直接在显存里 resize）
    //t3.data_gpu().resize(total_elements);

    int total_size = data_flat.size();
    gpu_permute_double_c_interface(
        data_flat.data(), // 源显存裸指针
        out.data(),   // 目标显存裸指针
        total_size,
        rank,
        d_order.data(),
        d_src_strides.data(),
        d_dst_dims.data(),
        d_dst_strides.data()
    );
}




struct BlockInfo {
    int pos;
    int size;
    int id; 
    std::string string( ) const{
        return std::format("BlockInfo: pos={}, size={}, id={}", pos, size, id);
    }
    bool operator==( const BlockInfo & other) const = default; 
}; 

using Type_ind_labels = std::variant<std::monostate, int, std::string>;   //std::monostate is used to check None value 


/* making std::variant printable by << */
template <typename T>
concept is_printable = requires(std::ostream& os, const T& val) {
    { os << val } -> std::same_as<std::ostream&>;
};
template <typename... Ts>
std::ostream& operator<<(std::ostream& os, const std::variant<Ts...>& var) {
    std::visit([&os](const auto& val) {
        using ValueType = std::decay_t<decltype(val)>;
        if constexpr (std::is_same_v<ValueType, std::monostate>) {
            os << "None";
        }
        else if constexpr (is_printable<ValueType>) {
            os << val;
        }
        else {
            os << "<not printable>";
        }
    }, var);
    return os;
}


void print_vec(const std::ranges::input_range auto vec, std::string name=""){
    if(name!=""){
        std::cout << name<<" = ";
    }
    for(auto v:vec){
        std::cout << v<<" ";
    }
    std::cout << "\n";
}




/*
    LEARN CPP: 
        about usage of concepts
            用法:
            concept 概念名 = 约束表达式;
        概念不是变量：concept 声明不定义对象或常量，而是定义一个judgement              
            concept的实质是以模板参量为参量的布尔函数；
            requires 的实质是编译期的assert；
        C++20 的概念机制完全建立在常量表达式之上。
        
        note1: using Has_Qn_t to tell the compliler that QspClass has Qn_t,  so that Qn_t can be refereed to 
        note2: the reason why there must be a 'typename': 
            the compiler by default takes QspClass:Qn_t as a variable or member
            function; typename is needed to let it know it is a class in the first
            place. 
*/

template <typename QspClass>
concept Has_Qn_t= requires {
    typename QspClass::Qn_t;   //note1
};
template <typename QspClass>
concept IsValidQsp = Has_Qn_t<QspClass> && std::derived_from<QspClass, 
        QspBase<QspClass, typename QspClass::Qn_t>>;  //note2
template<typename DTYPE>
concept IsNumber = std::is_same_v<DTYPE, float> || std::is_same_v<DTYPE, double> || std::is_same_v<DTYPE, std::complex<float>> || std::is_same_v<DTYPE, std::complex<double>>;

struct CpuDevice {}; 
struct GpuDevice {}; 

template <typename QspClass, typename DTYPE = double, typename Device=CpuDevice>
requires IsValidQsp<QspClass> && IsNumber<DTYPE>
//requires IsNumber<DTYPE>
class iTensor {
    public:
         
        using ThisType = iTensor<QspClass, DTYPE, Device>;  // 注入式类名 (Injected-class-name); 从而在声明 iTensor 类型时避免用避免用 CTAD （类模板参数推导，Class Template Argument Deduction）or write very long <...> 
        std::vector<QspClass> qsp_list;
        int rank;
            std::vector<Type_ind_labels> ind_labels;
            QspClass::Qn_t tot_qn=QspClass::Qn_t::qn_id() ; 
        
        int num_of_blocks=0;
        std::vector<BlockInfo> Block_idx; 
        std::vector<int> Addr_idx;  // of size num_of_blocks*rank; storing all qn_ind_tuple for non-zero blocks 
        std::vector<int> idx; 
        int tot_dim;
        std::vector<DTYPE> data_cpu;
        MyGpu::GpuData<DTYPE> data_gpu; 

            
        iTensor(std::vector<QspClass> qq): qsp_list(std::move(qq)) {
            rank =  qsp_list.size();
            set_data_entrance();
        }
        iTensor(std::vector<QspClass> qq, QspClass::Qn_t tqn): qsp_list(std::move(qq)), tot_qn(std::move(tqn)) {
            rank =  qsp_list.size();
            set_data_entrance();
        }
        
        static constexpr bool is_on_CPU(){
            return std::is_same_v<Device, CpuDevice>;
        }
        static constexpr bool is_on_GPU() {
            return std::is_same_v<Device, GpuDevice>;
        }
        
        void set_data_entrance() {
            /*
               learn cpp: 
               
               */
            num_of_blocks=0;
            Block_idx.clear();
            Addr_idx.clear();
            //idx.clear();
            
            
            std::vector<int> nqn_list;
            nqn_list.reserve(qsp_list.size()); 
            for(const auto& q : qsp_list) {nqn_list.push_back(q.nQN);}
            
            auto qn_ind_tuple_all = ndindex_F_order(nqn_list); 
            int num_of_qn_ind_tuple = static_cast<int>( qn_ind_tuple_all.size() / rank );
            
            idx.resize(num_of_qn_ind_tuple, -1);
            
            int* ptr = qn_ind_tuple_all.data(); 
            
            using Qn_t = typename decltype(qsp_list)::value_type::Qn_t;  // this line is only at compile time; 0 cost!; Good practice to use using everywhere.
            
            
            tot_dim=0;
            
            for(int qn_ind_tuple_id=0; qn_ind_tuple_id<num_of_qn_ind_tuple; qn_ind_tuple_id++){
                Qn_t  tqn = qsp_list[0].QNs[ptr[0]];
                for(int j = 1; j < rank; ++j) {
                    tqn += qsp_list[j].QNs[ptr[j]]; // 假设你定义了 operator+=
                    
                }
                //std::cout << "tqn: "<<tqn<<"\n";
                
                if(tqn==tot_qn){
                    int block_dim =1;
                    for(int j=0; j<rank; j++){ 
                        block_dim *= qsp_list[j].Dims[ptr[j]]; 
                        Addr_idx.push_back(ptr[j]);
                    }
                    Block_idx.push_back({tot_dim, block_dim, qn_ind_tuple_id}); 
                    idx[qn_ind_tuple_id] = num_of_blocks; 
                    
                    num_of_blocks+=1;
                    tot_dim += block_dim; 
                }
                
                ptr += rank;
            }
            if constexpr( is_on_CPU()){
                data_cpu.resize(tot_dim, 0.0);
            }
            else{
                data_gpu.resize(tot_dim);
            }
            
        }
        
        int get_block_id_by_qn_ind_tuple(const std::vector<int>& qn_ind_tuple) const {
            std::vector<int> dims;
            dims.reserve(rank);
            for(auto i=0; i<rank ; i++){dims.push_back(qsp_list[i].nQN);}
            int num = ravel_multi_index_F_order(qn_ind_tuple, dims);
            auto res = idx[num];
            assert(res != -1);   // invalid qn_ind_tuple
            return res;
        }
        
        //std::span<DTYPE> get_data_block(int block_id) {
        //    //*learn cpp*: 1. not not using "const",  span needed to be writable 2. using noexcept for compute intensive func
        //    const BlockInfo & block = Block_idx[block_id]; 
        //    return std::span<DTYPE>(data_cpu ).subspan(block.pos, block.size);
        //}
        
        //std::span<const DTYPE> get_data_block(int block_id) const noexcept{
        //    //*learn cpp*: 1. not not using "const",  span needed to be writable 2. using noexcept for compute intensive func
        //    const BlockInfo & block = Block_idx[block_id]; 
        //    return std::span<const DTYPE>(data_cpu).subspan(block.pos, block.size);
        //}
        auto get_data_block(int block_id){
            const BlockInfo & block = Block_idx[block_id]; 
            if constexpr(iTensor::is_on_CPU()){
                return std::span<DTYPE>(data_cpu ).subspan(block.pos, block.size); 
            }
            else{
                return MyGpu::GpuSpan<DTYPE>{data_gpu.data() + block.pos, block.size};
            }
        
        }
        
        auto get_data_block(int block_id) const {
            const BlockInfo & block = Block_idx[block_id]; 
            if constexpr(iTensor::is_on_CPU()){
                return std::span<const DTYPE>(data_cpu ).subspan(block.pos, block.size); 
            }
            else{
                return MyGpu::GpuSpan<const DTYPE>{data_gpu.data() + block.pos, block.size};
            }
        
        }
        
        
        std::vector<int> get_data_block_shape(int block_id) const noexcept{
            //qn_id_tuple = std::span(Addr_idx).subspan(block_id*rank, rank);
            const int *p = Addr_idx.data() + block_id*rank;
            std::vector<int> shape;
            //shape.reserve(rank); 
            for(int i=0; i<rank; i++){
                //std::cout <<qsp_list[i].Dims[p[i]] ;
                shape.push_back(qsp_list[i].Dims[p[i]]);
            }
            return  shape;
        }
        
        ThisType transpose(const std::vector<int>& order) const; 
        static std::tuple<std::vector<int>, std::vector<int>,std::vector<Type_ind_labels>,  std::vector<Type_ind_labels>> 
        prepare_leg(const std::vector<Type_ind_labels>& V1, const std::vector<Type_ind_labels>& V2) ;
        
        
        template <typename Device2>
        ThisType contract(const iTensor<QspClass, DTYPE,  Device2> & other) const; 

        std::string get_class_name() const {
                return demangle(typeid(*this).name());
            }        
        
            
};


template <typename QspClass, typename DTYPE, typename Device>
requires IsValidQsp<QspClass> && IsNumber<DTYPE>
iTensor<QspClass, DTYPE, Device> iTensor<QspClass, DTYPE, Device>::transpose(
        const std::vector<int>& order) const{
    std::vector<QspClass> qsp_list_new; 
    qsp_list_new.reserve(rank);
    for(auto i: order){
        qsp_list_new.push_back(qsp_list[i]);
    };
    //iTensor<QspClass, DTYPE, Device> res=iTensor(qsp_list_new, tot_qn);
    //auto res=iTensor(qsp_list_new, tot_qn);  // 模板参数推导 CTAD 可能失败
    //触发了 C++17 的 CTAD（类模板参数推导，Class Template Argument Deduction）
    //应该尽量避免  CTAD 
    //auto res=iTensor<QspClass, DTYPE, Device >(qsp_list_new, tot_qn);  // 模板参数推导可能失败
    
    auto res = ThisType(qsp_list_new, tot_qn );
    
    //std::cout << "this is on cpu: "<< this->is_on_CPU()<<"\n";
    //std::cout << "res is on cpu: "<< res.is_on_CPU()<<"\n";
    //std::cout <<"size: " << this->data_gpu.size()<<"\n";
    //std::cout <<"size: " << res.data_gpu.size()<<"\n";
    //std::cout << "name: " << get_class_name();
    //std::cout << "name: " << res.get_class_name();
    
    
    //std::vector<int> qn_ind_tuple;
    //qn_ind_tuple.reserve(rank);
    std::vector<int> qn_ind_tuple_new ;
    qn_ind_tuple_new.reserve(rank);
    
    const int * ptr; 
    
    for(auto block_id=0; block_id<num_of_blocks; block_id++){
        
        const std::vector<int> & shape = get_data_block_shape(block_id);
        auto data_block = get_data_block(block_id);
        
        // what is done next is to find the map  block_id -> block_id_new 
        qn_ind_tuple_new.clear();
        ptr = Addr_idx.data() + block_id*rank;
        for(auto i=0; i<rank; i++){
            //qn_ind_tuple.push_back(ptr[i]);
            qn_ind_tuple_new.push_back(ptr[order[i]]);
        }
        auto block_id_new = res.get_block_id_by_qn_ind_tuple(qn_ind_tuple_new);
        
        auto  data_block_new = res.get_data_block(block_id_new);
        
        //std::cout << "self type: "<< get_type_name(data_block)<<"\n";
        //std::cout << "res type: "<< get_type_name(data_block_new)<<"\n";
        
        
        if constexpr(iTensor::is_on_CPU()){
            transpose_dense<DTYPE>(data_block, shape, order, data_block_new);
        }
        else {
            transpose_dense_gpu(data_block, shape, order, data_block_new);
        }
        
        
    }
    
   return  res; 
}





template <typename QspClass, typename DTYPE, typename Device>
requires IsValidQsp<QspClass> && IsNumber<DTYPE>
std::tuple<std::vector<int>, std::vector<int>,std::vector<Type_ind_labels>,  std::vector<Type_ind_labels>> 
iTensor<QspClass, DTYPE, Device>::prepare_leg(const std::vector<Type_ind_labels>& V1, const std::vector<Type_ind_labels>& V2) 
{
    size_t rank1 = V1.size();
    size_t rank2 = V2.size();

    std::vector<int> Vp1(rank1);
    std::vector<int> Vp2(rank2);
    std::vector<Type_ind_labels> V3;
    std::vector<Type_ind_labels> leg_common;
    V3.reserve(rank1 + rank2);

    std::vector<bool> is_contracted_t1(rank1, false);
    std::vector<bool> is_contracted_t2(rank2, false);
    std::vector<std::pair<int, int>> matched_pairs;

    //  variant 已经重载了 == 算子
    for (size_t i = 0; i < rank1; ++i) {
        if (std::holds_alternative<std::monostate>(V1[i])) continue;
        
        for (size_t j = 0; j < rank2; ++j) {
            if (V1[i] == V2[j]) { 
                is_contracted_t1[i] = true;
                is_contracted_t2[j] = true;
                matched_pairs.push_back({static_cast<int>(i), static_cast<int>(j)});
                leg_common.push_back(V1[i]);

                //if (Dims1[i] != Dims2[j]) {
                //    throw std::runtime_error("error, dim of index to be contracted not equal!");
                //}
                break;
            }
        }
    }

    // 2. 构建 T1 秩序: [外腿, 内线]
    size_t k = 0;
    for (size_t i = 0; i < rank1; ++i) {
        if (!is_contracted_t1[i]) {
            Vp1[k++] = static_cast<int>(i);
            V3.push_back(V1[i]);
        }
    }
    for (const auto& pair : matched_pairs) Vp1[k++] = pair.first;

    // 3. 构建 T2 秩序: [内线, 外腿] （完美的 GEMM 连续对齐）
    size_t j = 0;
    for (const auto& pair : matched_pairs) Vp2[j++] = pair.second;
    
    for (size_t i = 0; i < rank2; ++i) {
        if (!is_contracted_t2[i]) {
            Vp2[j++] = static_cast<int>(i);
            V3.push_back(V2[i]);
        }
    }

    // Note the neccesity of std::move at here; do not confuse this with ROV; this calls the construction function of make_tuple
    return std::make_tuple(std::move(Vp1), std::move(Vp2), std::move(leg_common),  std::move(V3));
}    



template <typename QspClass, typename DTYPE, typename Device>
requires IsValidQsp<QspClass> && IsNumber<DTYPE>
template <typename Device2>
iTensor<QspClass, DTYPE, Device>   iTensor<QspClass, DTYPE, Device>::contract(
        const iTensor <QspClass, DTYPE, Device2>& other ) const {
    auto [ord1, ord2, ind_labels_internal,  ind_labels_3] = prepare_leg(ind_labels, other.ind_labels);
    static_assert(std::is_same_v<Device, Device2>, "两个张量必须在相同的硬件设备上才能进行收缩！");
    ThisType t1 = this->transpose(ord1);
    ThisType t2 = other.transpose(ord2);
    auto rank_1 = rank; 
    auto rank_2 = other.rank ;
    auto tot_qn_3 = tot_qn + other.tot_qn;
    int num_internal_legs = ind_labels_internal.size(); 
    int rank_3 = rank_1 + rank_2 - 2*num_internal_legs;
   
    std::cout << "rank_3="<<  rank_3 <<"\n";
    print_vec(ord1, "ord1 = ");
    print_vec(ord2, "ord2 = ");
    
    std::vector<QspClass> qsp_list_3;
    qsp_list_3.reserve(rank_3);
    for(int i=0; i<rank_1-num_internal_legs; i++){ 
        qsp_list_3.push_back(t1.qsp_list[i]); }
    for(int i=num_internal_legs; i<rank_2; i++){ 
        qsp_list_3.push_back(t2.qsp_list[i]); }
    
    iTensor<QspClass, DTYPE, Device> t3(qsp_list_3);
    
    t3.ind_labels = std::move(ind_labels_3);
    
    print_vec(t3.ind_labels, "t3.ind_labels");
    std::vector<int> qn_ind_tuple_1_internal; 
    qn_ind_tuple_1_internal.resize(num_internal_legs);
    std::vector<int> qn_ind_tuple_2_internal; 
    qn_ind_tuple_2_internal.resize(num_internal_legs);
    std::vector<int> qn_ind_tuple_3; qn_ind_tuple_3.resize(rank_3);
    int dim1, dim2, dimc; 
    int* ptr; 
    
    print_vec(t1.qsp_list, "t1.qsp_list");
    print_vec(t2.qsp_list, "t2.qsp_list");
    print_vec(t3.qsp_list, "t3.qsp_list");
    
    for(auto block_id_2=0; block_id_2<t2.num_of_blocks; block_id_2++ ){
        
        auto shape2 = t2.get_data_block_shape(block_id_2);
        
        ptr = t2.Addr_idx.data() + block_id_2*rank_2;
        dimc=1;
        for(auto i=0; i< num_internal_legs; i++) {
            dimc *= shape2[i];
            qn_ind_tuple_2_internal[i]=ptr[i];
        }
        
        dim2=1;
        for(auto i=num_internal_legs; i< t2.rank; i++) {
            dim2 *= shape2[i];
            qn_ind_tuple_3[i+num_internal_legs] = ptr[i]; 
        }
        
        
        auto data2 = t2.get_data_block(block_id_2);
       
        //Eigen::Map<Eigen::Matrix<const  DTYPE, Eigen::Dynamic, Eigen::Dynamic>> mat2(
        //        data2.data(), dimc, dim2);
        const Eigen::Map<Eigen::Matrix<DTYPE, 
              Eigen::Dynamic, Eigen::Dynamic>> mat2(data2.data(), dimc, dim2);
        //Eigen::Map<const Eigen::MatrixXd> mat2(data2.data(), dimc, dim2);

        
        for(auto block_id_1=0; block_id_1<num_of_blocks; block_id_1++) {
            ptr = t1.Addr_idx.data() + block_id_1*rank_1;
            
            for(auto i=0; i<num_internal_legs; i++){
                qn_ind_tuple_1_internal[i] = ptr[i + (rank_1-num_internal_legs)];
            }
            if(qn_ind_tuple_1_internal==qn_ind_tuple_2_internal){
                auto shape1 = t1.get_data_block_shape(block_id_1);
                
                dim1=1;
                for(auto i=0; i< rank - num_internal_legs; i++) { 
                    dim1 *= shape1[i]; 
                    qn_ind_tuple_3[i] = ptr[i];
                }
                auto data1 = t1.get_data_block(block_id_1);
                
                //Eigen::Map<Eigen::Matrix<const DTYPE, Eigen::Dynamic, Eigen::Dynamic>> mat1(data1.data(), dim1, dimc);
                //Eigen::Map<const Eigen::MatrixXd> mat1(data1.data(), dim1, dimc);
                const Eigen::Map<Eigen::Matrix<DTYPE, 
                      Eigen::Dynamic, Eigen::Dynamic>> mat1(data1.data(), dim1, dimc);
                //print_vec(qn_ind_tuple_3, "qn_ind_tuple_3");
                
                
                auto block_id_3 = t3.get_block_id_by_qn_ind_tuple(qn_ind_tuple_3);
                //std::cout << "block_id_3 = "<<block_id_3;
                auto data3 = t3.get_data_block(block_id_3);
                
                if constexpr(iTensor::is_on_CPU()){
                    Eigen::Map<Eigen::Matrix<DTYPE, 
                        Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> mat3(data3.data(), dim1, dim2);
                    mat3.noalias() = mat1 * mat2;  //  使用 .noalias() 告诉 Eigen 不存在内存别名（No Aliasing），即结果矩阵 C 的内存和 A、B 没有任何重叠。这允许 Eigen 略过中间临时变量的开销，直接把计算结果写回 span3，这是写高性能矩阵乘法的标准规范
                } else{
                    // 1. 严格检查并强转维度，防御 size_t 截断带来的巨坑
                    int M = static_cast<int>(dim1); // 矩阵 C 的行数 (A 的行数)
                    int N = static_cast<int>(dim2); // 矩阵 C 的列数 (B 的列数)
                    int K = static_cast<int>(dimc); // A 的列数 / B 的行数 (张量收缩的内指标维度)

                    const DTYPE alpha = 1.0;  //  准备GEMM 的系数： C = alpha * A * B + beta * C
                    const DTYPE beta  = 0.0;

                    const DTYPE* d_A = data1.data(); 
                    const DTYPE* d_B = data2.data();
                          DTYPE* d_C = data3.data();

                    // 计算主导维度（Leading Dimensions
                    // lda, b, c equals stride; for F order,  stride = num of rows of mat
                    int lda = M; 
                    int ldb = K;
                    int ldc = M; 
                    
                    //std::cout << "type data1: "<<get_type_name(data1) <<"\n"; 
                    //std::cout << "type data2: "<<get_type_name(data2) <<"\n"; 
                    //std::cout << "type data3: "<<get_type_name(data3) <<"\n"; 
                    //
                    //std::cout <<auto msg = std::format("dim1={},  dim2={},  dimc={}", dim1, dim2, dimc); <<"\n";
                    //std::cout << std::format("size1={}, size2={}, size3={}", data1.size(), data2.size(), data3.size())<<"\n";

                    // 5. 跨越编译器边界，向 RTX 3090 发射硬件算子！
                    if constexpr (std::is_same_v<DTYPE, double>) {
                    
                        //cudaError_t err_before = cudaDeviceSynchronize();
                        //if (err_before != cudaSuccess) {
                        //    std::cout << " 在进 GEMM 之前，GPU 就已经受过内伤了: " << cudaGetErrorString(err_before) << "\n";
                        //}            
                        //cudaPointerAttributes attrs_A;
                        //cudaPointerGetAttributes(&attrs_A, d_A);
                        //if (attrs_A.type == cudaMemoryTypeHost || attrs_A.type == cudaMemoryTypeUnregistered) {
                        //    std::cout << "🚨 【抓到现行】d_A 居然是一个 CPU 指针！cuBLAS 被你骗去读 CPU 内存了！\n";
                        //}            
                        
                        gpu_dgemm_c_interface(
                            M, N, K,
                            &alpha,
                            d_A, lda,
                            d_B, ldb,
                            &beta,
                            d_C, ldc
                        );
                        
                        //cudaError_t err_after = cudaDeviceSynchronize();
                        //if (err_after != cudaSuccess) {
                        //    std::cout << " GEMM 执行期间硬件暴毙: " << cudaGetErrorString(err_after) << "\n";
                        //}                        
                        

                    } else {
                        // 如果以后有 float 需求，可以在这里单加一个 gpu_sgemm_c_interface
                        static_assert(std::is_same_v<DTYPE, double>, "目前仅支持双精度 double 张量收缩");
                    }                    
                
                }
            }
            
        
        }
    
    }
    
    
    
    
    return  t3; 
}


namespace detail {
    // 完美的泛型解耦：TensorType 是一个鸭子类型
    template <typename TensorType>
    void set_data_entrance_impl(TensorType& self) {
        // 在这里，我们可以闭着眼睛用 self.is_on_CPU() 
        // 也可以用 self.data_gpu.resize()
        if constexpr (TensorType::is_on_CPU()) {
            self.data.resize(self.tot_dim, 0.0);
        } else {
            self.data_gpu.resize(self.tot_dim);
        }
    }
}



#ifdef TEST_IT
//#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

    TEST_CASE("ndindex") {
        std::vector<int> shape(3) ;
        shape={3, 3, 4};
        print_vec(shape); 
        
        auto xxx = ndindex_F_order(shape);
        std::cout <<xxx.size(); 
        //for(auto x:xxx){print_vec(x); 
        //    std::cout << "a \n";
        //}
        auto rank=shape.size();
        int *ptr=xxx.data(); 
        int len= static_cast<int>(xxx.size()/rank);
        
        for(int i=0; i<len;i++){
            std::span<int> chunk = std::span(xxx).subspan(i * rank, rank);
            //print_vec<std::span<int>>( chunk, "");
        }
        
        std::cout<<"\n";
    }
    


    TEST_CASE("set_data_entrance") {
        QspU1 qsp =  QspU1::easy_init({0, 1, -1}, {4, 3, 2}); 
        std::vector<QspU1> qsp_list;
        for(int i=0; i<3; i++){
            qsp_list.push_back(qsp);
        }
        
        iTensor<QspU1> t0(qsp_list);
        
        
        //t0.set_data_entrance(); 
        //print_vec<BlockInfo>(t0.Block_idx); 
        for(const BlockInfo & b : t0.Block_idx){
            std::cout <<b.string()<<"\n";
        }
        
        std::vector<int> vec(3);
      
        
        t0.get_data_block(1);
        print_vec(t0.get_data_block(1));
        
        std::cout << t0.Block_idx[4].string();
        CHECK(t0.Block_idx[4] == BlockInfo(136, 24, 15));
        
        auto v=t0.get_data_block_shape(1); 
        print_vec(v);
        std::vector<int> x={2, 3, 4};
        CHECK(v==x);
        
                
        std::cout<<"\n";
    }

    TEST_CASE("transpose") {
        QspU1 qsp =  QspU1::easy_init({0, 1, -1}, {2, 2, 2}); 
        std::vector<QspU1> qsp_list;
        for(int i=0; i<3; i++){
            qsp_list.push_back(qsp);
        }
        
        iTensor<QspU1> t0(qsp_list);
        for(auto i=0; i<t0.data_cpu.size(); i++){
            t0.data_cpu[i]=static_cast<double>(i);
        }
        print_vec(t0.data_cpu);

        
        std::vector<int> order={0, 2, 1};
        iTensor<QspU1, double> tp = t0.transpose(order);
        
        print_vec(tp.data_cpu);
        
        std::vector<double> old = {0, 1, 4, 5, 2, 3, 6, 7, 24, 25, 28, 29, 26, 27, 30, 31, 
            40, 41, 44, 45, 42, 43, 46, 47, 8, 9, 12, 13, 10, 
            11, 14, 15, 48, 49, 52, 53, 50, 51, 54, 55, 16, 17, 
            20, 21, 18, 19, 22, 23, 32, 33, 36, 37, 34, 35, 38, 39};
        CHECK(tp.data_cpu == old);
        
        
        using iTensor_gpu = iTensor<QspU1, double,  GpuDevice>;
        auto qsp_list_1 = qsp_list ; 
        iTensor_gpu  t_gpu(qsp_list_1);
        std::cout << "size of data_gpu: " <<  t_gpu.data_gpu.size();
        t_gpu.ind_labels={"a", "b", "c"};
        
        std::vector<double> dd;
        dd.resize(t_gpu.data_gpu.size());
        for(auto i=0; i<dd.size(); i++){ dd[i]=static_cast<double>(i); }
        t_gpu.data_gpu.copy_from_host(dd);
        auto t_gpu_trans = t_gpu.transpose({0, 2, 1});
        auto ee = t_gpu_trans.data_gpu.to_vector();
        print_vec(ee, "ee= "); 
        
        CHECK(ee == old);
        
        
        
        
                
        std::cout<<"\n";
    }


    TEST_CASE("temp") {
        QspU1 qsp =  QspU1::easy_init({0, 1, -1}, {2, 2, 2}); 
        std::vector<QspU1> qsp_list;
        for(int i=0; i<3; i++){
            qsp_list.push_back(qsp);
        }
        QspU1 qsp_2 =  QspU1::easy_init({0, -1, 1}, {2, 2, 2}); 
        std::vector<QspU1> qsp_list_2;
        for(int i=0; i<3; i++){
            qsp_list_2.push_back(qsp_2);
        }
        
        using DTYPE = double ;
        
        std::vector<DTYPE> dd; 
        dd.resize(56);
        for(auto i=0; i<dd.size(); i++){ dd[i]=static_cast<double>(i); }
        
        
        std::vector<DTYPE> old=  {2, 3, 6, 7, 6, 11, 26, 31, 10, 19, 46, 55, 14, 27, 66, 79, 26, 27, 30, 31, 126, 131, 146, 151, 226, 235, 262, 271, 326, 339};
        
        // cpu version 
        {
            iTensor<QspU1> t0(qsp_list);
            for(auto i=0; i<t0.data_cpu.size(); i++){ t0.data_cpu[i]=static_cast<double>(i); }
            t0.ind_labels={"a", "b", "c"};
            print_vec(t0.ind_labels, "t0.ind_labels");
            
            iTensor<QspU1> t2(qsp_list_2);
            t2.ind_labels={"b", "e", "f"};

            for(auto i=0; i<t2.data_cpu.size(); i++){ t2.data_cpu[i]=static_cast<double>(i); }
                
            print_vec(t2.ind_labels, "t2.ind_labels");
            
            
            auto t3 = t0.contract(t2);
            print_vec(t3.data_cpu);
            std::vector<double> temp(t3.data_cpu.data(), t3.data_cpu.data()+30 );
            CHECK(temp==old);
        }
        

        QspU1 qsp_1 =  QspU1::easy_init({0, 1, -1}, {2, 2, 2}); 
        std::vector<QspU1> qsp_list_1;
        for(int i=0; i<3; i++){
            qsp_list_1.push_back(qsp_1);
        }
        
        
        using iTensor_gpu = iTensor<QspU1, DTYPE,  GpuDevice>;
        iTensor_gpu  t0(qsp_list_1);
        
        //for(auto i=0; i<t0.data_cpu.size(); i++){ t0.data_cpu[i]=static_cast<double>(i); }
        std::cout << "size of data_gpu: " <<  t0.data_gpu.size();
        t0.ind_labels={"a", "b", "c"};
        
        //std::vector<double> dd;
        //dd.resize(t0.data_gpu.size());
        //for(auto i=0; i<dd.size(); i++){ dd[i]=static_cast<double>(i); }
        t0.data_gpu.copy_from_host(dd);
        auto t0_trans = t0.transpose({0, 2, 1});
        auto ee = t0_trans.data_gpu.to_vector();
        print_vec(ee, "ee= "); 
        
        iTensor_gpu  t2(qsp_list_2);
        t2.data_gpu.copy_from_host(dd);
        t2.ind_labels={"b", "e", "f"};
        auto t3 = t0.contract(t2);
        
        auto xx11 = t3.data_gpu.to_vector();
        std::vector<DTYPE> temp(xx11.data(), xx11.data()+30);
        CHECK(temp==old);
       
                
        std::cout<<"\n";
    }
   

#endif






