#include <iterator>
#include <tuple>
#include <vector>
#include <algorithm>
#include <iostream>
#include "omp.h"
#include <cstdlib>
#include <memory>
#include <atomic>
#include <concepts>

constexpr int BLOCK_SIZE = 128;

template <typename T>
class Matrix
{
    int rows_num_ = 0;
    int cols_num_ = 0;

    std::vector<T> content_;
public:
    explicit Matrix(int rows = 1, int cols = 1) : rows_num_(rows), cols_num_(cols), content_(rows * cols, T{})
    {}

    static Matrix rand_mat(int rows = 1, int cols = 1)
    {
        Matrix ret(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                ret(i, j) = ::rand() % 10;
                // std::cout << ret(i, j) << " ";
            }
        // std::cout << std::endl;
        }
        return ret;
    }

    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    using value_type = T;

    iterator begin () {return content_.begin();}
    iterator end () {return content_.end();}

    template <typename it>
    requires std::is_same_v<Matrix, typename std::iterator_traits<it>::value_type>
    static Matrix from2x2(it mat_begin, it mat_end)
    {
        auto&& rows = mat_begin->rows_num_;
        auto&& cols = mat_begin->rows_num_;

        auto&& fourth = (rows * cols);

        auto mat1 = mat_begin;
        auto mat2 = ++mat_begin;
        auto mat3 = ++mat_begin;
        auto mat4 = ++mat_begin;

        Matrix ret{rows * 2, cols * 2};

        // std::cout << "++++++++++" << std::endl;
        // mat1->print();
        // mat2->print();
        // mat3->print();
        // mat4->print();

        // std::cout << "++++++++++" << std::endl;

        #pragma omp parallel for simd collapse(2)   
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
            {
                ret(i, j) = mat1->at(i, j);
                ret(i, j + cols) = mat2->at(i, j);
                ret(i + rows, j) = mat3->at(i, j);
                ret(i + rows, j + cols) = mat4->at(i, j);
            }

        return ret;
    }


    std::array<Matrix, 4> split2x2() const
    {
        std::array<Matrix, 4> matrix_arr;
       
        auto&& new_row_size = rows_num_ / 2;
        auto&& new_col_size = cols_num_ / 2;

        for (auto&& mat_it : matrix_arr)
            mat_it = Matrix(new_row_size, new_col_size);
       
        #pragma omp parallel for simd collapse(2)   
        for (int i = 0; i < new_row_size; i++)
            for (int j = 0; j < new_col_size; j++)
            {
                matrix_arr[0](i, j) = at(i, j);
                matrix_arr[1](i, j) = at(i, j + new_col_size);
                matrix_arr[2](i, j) = at(i + new_row_size, j);
                matrix_arr[3](i, j) = at(i + new_row_size, j + new_col_size);
            }
        return matrix_arr;
    }

    int cols() const {return cols_num_;}
    int rows() const {return rows_num_;}

    T& operator()(int row, int col) {return at(row, col); }
    const T& operator()(int row, int col) const {return at(row, col);}
     
    T& at(int row, int col) {return content_[row * cols_num_ + col];}
    const T& at(int row, int col) const {return content_[row * cols_num_ + col];}

    bool operator==(const Matrix& other) const { return content_ == other.content_; }

    void transpose_in_place() 
    {
        for (int i = 0; i < rows_num_; i++)
            for (int j = i; j < cols_num_; j++)
            {
                std::swap(at(i, j), at(j, i));
            }
    }

    Matrix transpose() const
    {
        Matrix transposed{*this};
        
        transposed.transpose_in_place();
        return transposed;
    }


    void print() const noexcept
    {
        for (int i = 0; i < rows_num_; i++)
        {
            for (int j = 0; j < rows_num_; j++)
                std::cout << at(i, j) << " ";
            std::cout << std::endl;
        }
    }


    Matrix& add (const Matrix& rhs)
    {
        auto size = content_.size();
        #pragma omp parallel for simd   
        for (int i = 0; i < size; i++)
            content_[i] = content_[i] + rhs.content_[i];
        
        return *this;
    }

    Matrix& sub (const Matrix& rhs)
    {
        auto size = content_.size();
        #pragma omp parallel for simd   
        for (int i = 0; i < size; i++)
            content_[i] = content_[i] - rhs.content_[i];
        
        return *this;
    }
};

template <typename T>
Matrix<T> operator+(const Matrix<T>& lhs, const Matrix<T>& rhs)
{
    Matrix tmp{lhs};
    return tmp.add(rhs);
}

    template <typename T>
Matrix<T> operator-(const Matrix<T>& lhs, const Matrix<T>& rhs)
{
    Matrix tmp{lhs};
    return tmp.sub(rhs);
}



template <typename T>
Matrix<T> basic_mult(const Matrix<T>& A, const Matrix<T>& B)
{
    auto A_rows = A.rows();
    auto A_cols = A.cols();
    auto B_cols = B.cols();

    Matrix<T> C(A_rows, B_cols);

    T tmp = 0;
    for (int i = 0; i < A_rows; i++)
        for (int j = 0; j < B_cols; j++)
        {
            for (int k = 0; k < A_cols; k++)
                {
                    tmp += A(i, k) * B(k, j);
                }
            C(i, j) = tmp;
            tmp = 0;
        }

    return C;
}

template <typename T>
Matrix<T> basic_par_mult(const Matrix<T>& A, const Matrix<T>& B)
{
    auto A_rows = A.rows();
    auto A_cols = A.cols();
    auto B_cols = B.cols();

    Matrix<T> C(A_rows, B_cols);

    T tmp = 0;
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < A_rows; i++)
        for (int j = 0; j < B_cols; j++)
        {
            for (int k = 0; k < A_cols; k++)
                {
                    tmp += A(i, k) * B(k, j);
                }
            C(i, j) = tmp;
            tmp = 0;
        }

    return C;
}

template <typename T>
Matrix<T> optimized_mult(const Matrix<T>& A, const Matrix<T>& B)
{
    auto A_rows = A.rows();
    auto A_cols = A.cols();
    auto B_cols = B.cols();
    auto transposed_B =  B.transpose();

    Matrix<T> C(A_rows, B_cols);

    for (int block_j = 0; block_j < B_cols; block_j += BLOCK_SIZE)
    {
        auto max_block_j = std::min(block_j + BLOCK_SIZE, B_cols);
        for (int block_k = 0; block_k < A_cols; block_k += BLOCK_SIZE)
            {
                auto max_block_k = std::min(block_k + BLOCK_SIZE, A_cols);
                for (int i = 0; i < A_rows; i++)
                    for (int j = block_j; j < max_block_j; j++)
                    {
                        T tmp = 0;
                        #pragma omp simd
                        for (int k = block_k; k < max_block_k; k++)
                            {
                                tmp += A(i, k) * transposed_B(j, k);
                                // std::cout << "access at " << i << " " << j << " " << k << std::endl;
                            }
                        C(i, j) += tmp;
                    }
            }
    }
    return C;
}

template <typename T>
Matrix<T> optimized_par_mult(const Matrix<T>& A, const Matrix<T>& B)
{
    auto A_rows = A.rows();
    auto A_cols = A.cols();
    auto B_cols = B.cols();
    auto transposed_B =  B.transpose();

    Matrix<T> C(A_rows, B_cols);

    #pragma omp parallel for collapse(2)
    for (int block_j = 0; block_j < B_cols; block_j += BLOCK_SIZE)
    {
        for (int block_k = 0; block_k < A_cols; block_k += BLOCK_SIZE)
            {
                auto max_block_j = std::min(block_j + BLOCK_SIZE, B_cols);
                auto max_block_k = std::min(block_k + BLOCK_SIZE, A_cols);
                for (int i = 0; i < A_rows; i++)
                    for (int j = block_j; j < max_block_j; j++)
                    {
                        T tmp = 0;
                        #pragma omp simd
                        for (int k = block_k; k < max_block_k; k++)
                            {
                                tmp += A(i, k) * transposed_B(j, k);
                                // std::cout << "access at " << i << " " << j << " " << k << std::endl;
                            }
                        C(i, j) += tmp;
                    }
            }
    }
    return C;
}


template <typename T>
Matrix<T> strassen_mult(const Matrix<T>& A, const Matrix<T>& B)
{
    auto A_rows = A.rows();
    auto A_cols = A.cols();
    auto B_cols = B.cols();
    if (A_rows <= 64)
    //if (A_rows <= 1)
        return optimized_mult(A, B);


    auto&& [A1, A2, A3, A4] = A.split2x2();
    auto&& [B1, B2, B3, B4] = B.split2x2();

    std::array<Matrix<T>, 7> M;
    M[0] = strassen_mult(A1 + A4, B1 + B4);
    M[1] = strassen_mult(A3 + A4, B1);
    M[2] = strassen_mult(A1, B2 - B4);
    M[3] = strassen_mult(A4, B3 - B1);
    M[4] = strassen_mult(A1 + A2, B4);
    M[5] = strassen_mult(A3 - A1, B1 + B2);
    M[6] = strassen_mult(A2 - A4, B3 + B4);

    std::array<Matrix<T>, 4> C_submat;

    C_submat[0] = M[0] + M[3] - M[4] + M[6];
    C_submat[1] = M[2] + M[4];
    C_submat[2] = M[1] + M[3];
    C_submat[3] = M[0] - M[1] + M[2] + M[5];

    return Matrix<T>::from2x2(C_submat.begin(), C_submat.end());
}


template <typename T>
Matrix<T> par_strassen_mult(const Matrix<T>& A, const Matrix<T>& B)
{
    auto A_rows = A.rows();
    auto A_cols = A.cols();
    auto B_cols = B.cols();
    if (A_rows <= 16)
    //if (A_rows <= 1)
        return optimized_par_mult(A, B);


    auto&& [A1, A2, A3, A4] = A.split2x2();
    auto&& [B1, B2, B3, B4] = B.split2x2();

    std::array<Matrix<T>, 7> M;
    #pragma omp parallel sections
    {
        #pragma omp section
        M[0] = strassen_mult(A1 + A4, B1 + B4);
        #pragma omp section
        M[1] = strassen_mult(A3 + A4, B1);
        #pragma omp section
        M[2] = strassen_mult(A1, B2 - B4);
        #pragma omp section
        M[3] = strassen_mult(A4, B3 - B1);
        #pragma omp section
        M[4] = strassen_mult(A1 + A2, B4);
        #pragma omp section
        M[5] = strassen_mult(A3 - A1, B1 + B2);
        #pragma omp section
        M[6] = strassen_mult(A2 - A4, B3 + B4);
    }


    std::array<Matrix<T>, 4> C_submat;
    #pragma omp parallel sections
    {
        #pragma omp section
        C_submat[0] = M[0] + M[3] - M[4] + M[6];
        #pragma omp section
        C_submat[1] = M[2] + M[4];
        #pragma omp section
        C_submat[2] = M[1] + M[3];
        #pragma omp section
        C_submat[3] = M[0] - M[1] + M[2] + M[5];
    }
    return Matrix<T>::from2x2(C_submat.begin(), C_submat.end());
}

