#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <algorithm> 

template <typename Derived>   //this is CRTP (Curiously Recurring Template Pattern) fashion
//ATTENTION: one cannot istancialize a template class 
class QnBase {
	public:
        int val; 
        const std::string SYMMETRY; 
        QnBase() : val(0) {}
        QnBase(int v) : val(v) {}
        
        
        // & gurantees passing name while not value                                                
        bool operator==(const Derived& other) const{
            return this->val==other.val;
            }
        
        QnBase operator+(const Derived& other) const {
            return QnBase{this->val+other.val} ;
            } 
        
        // 友元重载 <<
        friend std::ostream& operator<<(std::ostream& os, const Derived& qn) {
            os << "("<<qn.val<<")";
            return os;
            }
        
        //virtual void displayInfo() const {
        //std::cout << "Tensor Name: " << name << std::endl;                                                 
        //}
    private:
};

class QnTravial: public QnBase<QnTravial> {
    public:
        int val=1;
        const std::string SYMMETRY="Travial";
        using QnBase::QnBase; 
        
        void reverse(){}
        
        QnTravial conj()
        {
            return *this; 
        
        }
        QnTravial operator+(const QnTravial& other)
        {
            return *this;
        
        }
};

class QnU1:public QnBase<QnU1> {
    public:
        //int val;  //会造成name shadowing with base class 
        //
        using QnBase::val;  // 增加透明性
        const std::string SYMMETRY="U1"; 
        //QnU1() = default ;
        using QnBase::QnBase;
        
        
        QnU1(int v) : QnBase(v) {}
        QnU1 operator+(const QnU1& other) const {
            return QnU1{this->val+other.val} ;
            } 
        
        void reverse() { this->val=-this->val; }
        
        QnU1 conj() const{
            return QnU1{-this->val}; 
            }
        
        static QnU1 qn_id() {
            return QnU1{0};
            }
        
        QnU1 copy() const {
            return QnU1{this->val}; 
            }
    private:
}; 


template <typename Derived, typename QnClass>
class QspBase {
    public:
        int nQN=0;
        std::vector<QnClass> QNs; 
        std::vector<int> Dims; 
        
        mutable int totDim=0;  // for lazy evaluation; using mutable, so that can be modified by const function
        //using QnClass = QnBase;
        
        
        //base class construction func should not using Derived
        QspBase(int n_val, std::vector<QnClass> qns_val, std::vector<int> dims_val):
            nQN(n_val), QNs(std::move(qns_val)), Dims(std::move(dims_val)) {}
        
        static Derived null() {
            // qn_id() 对应 QnBase 的默认构造（通常 val=0）
            QnClass qn = QnClass::qn_id(); 
            return Derived(1, {qn}, {1});
        }
        
        bool operator==(const Derived& other) const {
            //std::cout << "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"<<nQN<<totDim<<other.totDim;
            if (nQN != other.nQN) {
                return false;
                }
            if (this->Dims != other.Dims) {
                return false;
                }
            
            if (this->QNs != other.QNs) {
                return false;
                }
            
            return true;
        }
        
        int get_tot_dim() const {
            if(totDim!=0) {
                return this->totDim;
            }
            else {
                int sum = 0;
                for (const auto& d : Dims) sum += d;
                this->totDim = sum;
                return this->totDim; 
            }
        }
        
        //Derived operator*(const Derived& other) const{
        //}
        
        // Checks if a quantum number exists.  Returns the index if found, otherwise -1.
        int has_quant_num(const QnClass& qn) const {
            auto pos = std::find(this->QNs.begin(), this->QNs.end(), qn);
            if (pos != this->QNs.end()) {
                auto ind = std::distance(this->QNs.begin(), pos); 
                return static_cast<int>(ind);
            }
            return -1;
        }

        // Direct sum of vector spaces subject to symmetry.
        void add_to_quant_space(const QnClass& qn, int d) {
            int i = has_quant_num(qn);
            // In HPC, we use assertions for sanity checks that disappear in release builds
            // Note: Ensure MaxQNNum is defined in your class or passed as a constraint
            // assert(i < this->MaxQNNum); 

            if (i < 0) {
                this->QNs.push_back(qn);
                this->Dims.push_back(d);
                this->nQN++;
            } else {
                this->Dims[i] += d;
            }
            this->totDim += d;
        }

        // Performs a tensor product of two quantum spaces.
        Derived tensor_prod(const Derived& other) const {
            // Handle identity cases (nQN == 0)
            if (this->nQN == 0) return other; 
            if (other.nQN == 0) return *static_cast<const Derived*>(this);  //this 虽然指向派生类（如 QspU1），但它的静态类型是基类指针; 需要将基类指针显式转换为派生类指针

            // Initialize result using the derived type
            // We use Derived instead of QspBase to ensure the correct object is returned
            //Derived res = Derived::null(); 
            Derived res{0, {}, {}}; 
            res.totDim = 0;

            for (int i = 0; i < this->nQN; ++i) {
                for (int j = 0; j < other.nQN; ++j) {
                    QnClass qn = this->QNs[i] + other.QNs[j];
                    int d = this->Dims[i] * other.Dims[j];
                    res.add_to_quant_space(qn, d);
                }
            }
            return res;
        }
        
        static Derived easy_init(std::vector<int> qns,  std::vector<int> dims) {
            int nqn=qns.size();
            std::vector<QnClass> QNs; 
            QNs.reserve(nqn); 
            std::transform(qns.begin(), qns.end(), std::back_inserter(QNs), 
                    [](int x) { return QnClass(x); }
                    );
            return Derived{nqn, QNs, dims};

        }    
        
        //ATTENTION: KEEP this.  in cpp, copy is not needed any more
        //Derived copy() const {
        //    return Derived{nQN, QNs, Dims};
        //}
        
        //友元<<重载 
        friend std::ostream& operator<<(std::ostream& os, const Derived& qsp) {
            for(int i=0;i<qsp.nQN;i++)
            {
                os << qsp.QNs[i]<<qsp.Dims[i];
                if(i<qsp.nQN-1) { os<<"+"; }
            }
            return os;
            }
};

class QspTravial: public QspBase<QspTravial, QnTravial> {
    public:
        //int nQN;     // attention: should not redefine nQN, etc.
        //std::vector<QnTravial> QNs; 
        //std::vector<int> Dims; 
        
        using QspBase::QspBase;   // this is nuclear weapon 
        
        //using QnClass = QnTravial;
        QspTravial(int n, std::vector<QnTravial> q, std::vector<int> d) 
                : QspBase(n, q, d) {} 
    private:
};

class QspU1 : public QspBase<QspU1, QnU1> {
    public:
        using QspBase::QspBase;  // 继承基类的构造逻辑
        // 子类需要提供对应的构造函数
        QspU1(int n, std::vector<QnU1> q, std::vector<int> d) 
            : QspBase(n, q, d) {}
};



/*/#ifdef TEST_IT
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
    
    TEST_CASE("test qn") {
        // 这里写你的测试逻辑
        QnU1 qn0{0}, qn1{1}, qnm1{-1}; 
        std::cout<< qn0<< qn1;
        CHECK(qn1+qnm1==qn0);
        std::cout<<"\n";
    }

    TEST_CASE("test qspu1") {
        // 这里写你的测试逻辑
        QspU1 qsp0 =  QspU1::easy_init({0, 1, -1}, {1, 1, 1}); 
        QspU1 qsp2{5, {0, 1, -1, 2, -2}, {3, 2, 2, 1, 1}};
        
        std::cout << qsp0.tensor_prod(qsp0) <<"\n";
        std::cout << qsp2;
        
        CHECK(qsp2==qsp0.tensor_prod(qsp0));

        //CHECK(result.nQN == 2);
        // ... 其他 CHECK ...
        std::cout<<"\n";
    }

    TEST_CASE("temp") {
        // 这里写你的测试逻辑
        QnU1 qn0{0}, qn1{1}, qnm1{-1}; 
        QspU1 qsp0 =  QspU1::easy_init({0, 1, -1}, {1, 1, 1}); 
        std::cout << qsp0.tensor_prod(qsp0);
        
        
        std::cout<<"\n";
    }

//#endif
*/




