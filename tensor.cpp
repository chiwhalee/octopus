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
#include <algorithm> 
#include <variant>  // std::variant 
#include <span>  //std::span  // chunck of mem window 
#include <Eigen/Dense>                 
#include <Eigen/Core>

//#include <format>
#include <cassert>

#include <concepts>
#include <type_traits>

#include "utilities.cpp"
#include "quantum_number.cpp"





void ndindex(const std::vector<int>& shape) {
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


std::vector<int> ndindex_F_order(const std::vector<int>& shape) {
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


// equivalent to np.ravel_multi_index for 'F' order
int ravel_multi_index_F_order(const std::vector<int>& indices, const std::vector<int>& dims) {
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
//            if (++it_idx[r] < shape[r]) break;
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

    std::vector<int> it_idx(rank, 0);
    std::vector<int> out_idx(rank); // 在循环外分配空间
    int total_size = array_flat.size();
    
    for (int i = 0; i < total_size; ++i) {
        // 映射索引
        for(int r=0; r<rank; ++r) out_idx[r] = it_idx[order[r]];
        
        // i 就是 array_flat 的平铺索引，无需再次 ravel
        int out_flat = ravel_multi_index_F_order(out_idx, new_shape);
        out[out_flat] = array_flat[i];

        // 更新 Fortran order 迭代器
        for (int r = 0; r < rank; ++r) {
            if (++it_idx[r] < shape[r]) break;
            it_idx[r] = 0;
        }
    }
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

template <typename QspClass, typename DTYPE = double>
requires IsValidQsp<QspClass> && IsNumber<DTYPE>
//requires IsNumber<DTYPE>
class iTensor {
    public:
        std::vector<QspClass> qsp_list;
        int rank;
            std::vector<Type_ind_labels> ind_labels;
            QspClass::Qn_t tot_qn=QspClass::Qn_t::qn_id() ; 
        
        int num_of_blocks=0;
        std::vector<BlockInfo> Block_idx; 
        std::vector<int> Addr_idx;  // of size num_of_blocks*rank; storing all qn_ind_tuple for non-zero blocks 
        std::vector<int> idx; 
        int tot_dim;
        std::vector<DTYPE> data;
            
        iTensor(std::vector<QspClass> qq): qsp_list(std::move(qq)) {
            rank =  qsp_list.size();
            set_data_entrance();
        }
        iTensor(std::vector<QspClass> qq, QspClass::Qn_t tqn): qsp_list(std::move(qq)), tot_qn(std::move(tqn)) {
            rank =  qsp_list.size();
            set_data_entrance();
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
            data.resize(tot_dim, 0.0);
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
        
        std::span<DTYPE> get_data_block(int block_id) {
            //*learn cpp*: 1. not not using "const",  span needed to be writable 2. using noexcept for compute intensive func
            const BlockInfo & block = Block_idx[block_id]; 
            return std::span<DTYPE>(data ).subspan(block.pos, block.size);
        }
        
        std::span<const DTYPE> get_data_block(int block_id) const noexcept{
            //*learn cpp*: 1. not not using "const",  span needed to be writable 2. using noexcept for compute intensive func
            const BlockInfo & block = Block_idx[block_id]; 
            return std::span<const DTYPE>(data ).subspan(block.pos, block.size);
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
        
        iTensor<QspClass, DTYPE> transpose(const std::vector<int>& order) const{
            std::vector<QspClass> qsp_list_new; 
            qsp_list_new.reserve(rank);
            for(auto i: order){
                qsp_list_new.push_back(qsp_list[i]);
            };
            iTensor<QspClass,  DTYPE> res=iTensor(qsp_list_new, tot_qn);
            
            //std::vector<int> qn_ind_tuple;
            //qn_ind_tuple.reserve(rank);
            std::vector<int> qn_ind_tuple_new ;
            qn_ind_tuple_new.reserve(rank);
            
            const int * ptr; 
            
            for(auto block_id=0; block_id<num_of_blocks; block_id++){
                
                const std::vector<int> & shape = get_data_block_shape(block_id);
                std::span<const DTYPE> data_block = get_data_block(block_id);
                
                // what is done next is to find the map  block_id -> block_id_new 
                qn_ind_tuple_new.clear();
                ptr = Addr_idx.data() + block_id*rank;
                for(auto i=0; i<rank; i++){
                    //qn_ind_tuple.push_back(ptr[i]);
                    qn_ind_tuple_new.push_back(ptr[order[i]]);
                }
                auto block_id_new = res.get_block_id_by_qn_ind_tuple(qn_ind_tuple_new);
                
                
                std::span<DTYPE> data_block_new = res.get_data_block(block_id_new);
                transpose_dense<DTYPE>(data_block, shape, order, data_block_new);
                
            }
            
           return  res; 
        }
        
        
        std::tuple<std::vector<int>, std::vector<int>,std::vector<Type_ind_labels>,  std::vector<Type_ind_labels>> 
        prepare_leg(const std::vector<Type_ind_labels>& V1, const std::vector<Type_ind_labels>& V2) 
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
        
        iTensor<QspClass, DTYPE> contract(const iTensor<QspClass, DTYPE> other ){
            auto [ord1, ord2, ind_labels_internal,  ind_labels_3] = this->prepare_leg(this->ind_labels, other.ind_labels);
            
            
            auto t1 = this->transpose(ord1);
            auto t2 = other.transpose(ord2);
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
            iTensor<QspClass, DTYPE> t3(qsp_list_3);
            
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
                        print_vec(qn_ind_tuple_3, "qn_ind_tuple_3");
                        
                        
                        auto block_id_3 = t3.get_block_id_by_qn_ind_tuple(qn_ind_tuple_3);
                        //std::cout << "block_id_3 = "<<block_id_3;
                        auto data3 = t3.get_data_block(block_id_3);
                        Eigen::Map<Eigen::Matrix<DTYPE, 
                            Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat3(data3.data(), dim1, dim2);
                        mat3.noalias() = mat1 * mat2;
                    }
                    
                
                }
            
            
            }
            
            
            
            
            return  t3; 
        }
            
};


  



//#ifdef TEST_IT
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
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
        for(auto i=0; i<t0.data.size(); i++){
            t0.data[i]=static_cast<double>(i);
        }
        print_vec(t0.data);

        
        std::vector<int> order={0, 2, 1};
        iTensor<QspU1, double> tp = t0.transpose(order);
        
        print_vec(tp.data);
        
        std::vector<double> old = {0, 1, 4, 5, 2, 3, 6, 7, 24, 25, 28, 29, 26, 27, 30, 31, 
            40, 41, 44, 45, 42, 43, 46, 47, 8, 9, 12, 13, 10, 
            11, 14, 15, 48, 49, 52, 53, 50, 51, 54, 55, 16, 17, 
            20, 21, 18, 19, 22, 23, 32, 33, 36, 37, 34, 35, 38, 39};
        CHECK(tp.data == old);
        
        
                
        std::cout<<"\n";
    }


    TEST_CASE("temp") {
        QspU1 qsp =  QspU1::easy_init({0, 1, -1}, {2, 2, 2}); 
        std::vector<QspU1> qsp_list;
        for(int i=0; i<3; i++){
            qsp_list.push_back(qsp);
        }
        
        iTensor<QspU1> t0(qsp_list);
        for(auto i=0; i<t0.data.size(); i++){ t0.data[i]=static_cast<double>(i); }
        t0.ind_labels={"a", "b", "c"};
        print_vec(t0.ind_labels, "t0.ind_labels");
        
        
        QspU1 qsp_2 =  QspU1::easy_init({0, -1, 1}, {2, 2, 2}); 
        std::vector<QspU1> qsp_list_2;
        for(int i=0; i<3; i++){
            qsp_list_2.push_back(qsp_2);
        }
        
        iTensor<QspU1> t2(qsp_list_2);
        t2.ind_labels={"b", "e", "f"};

        for(auto i=0; i<t2.data.size(); i++){ t2.data[i]=static_cast<double>(i); }
            
        print_vec(t2.ind_labels, "t2.ind_labels");
        
        
        auto t3 = t0.contract(t2);
        print_vec(t3.data);
        //for(auto label: t3.ind_labels){
        //    std::cout << label << "a";
        //}
        
        
                
        std::cout<<"\n";
    }
   

//#endif






