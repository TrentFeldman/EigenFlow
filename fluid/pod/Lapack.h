// Fortran LAPACK/BLAS prototypes (column-major). Trailing size_t args are the hidden
// character-length parameters gfortran-built libraries expect; MKL/OpenBLAS ignore them.
#pragma once
#include <cstddef>

extern "C" {
void dgesdd_(const char* jobz, const int* m, const int* n, double* a, const int* lda, double* s, double* u,
             const int* ldu, double* vt, const int* ldvt, double* work, const int* lwork, int* iwork, int* info,
             std::size_t jobz_len);
void dsyevd_(const char* jobz, const char* uplo, const int* n, double* a, const int* lda, double* w, double* work,
             const int* lwork, int* iwork, const int* liwork, int* info, std::size_t jobz_len, std::size_t uplo_len);
void dsyrk_(const char* uplo, const char* trans, const int* n, const int* k, const double* alpha, const double* a,
            const int* lda, const double* beta, double* c, const int* ldc, std::size_t uplo_len, std::size_t trans_len);
void dgemm_(const char* transa, const char* transb, const int* m, const int* n, const int* k, const double* alpha,
            const double* a, const int* lda, const double* b, const int* ldb, const double* beta, double* c,
            const int* ldc, std::size_t ta_len, std::size_t tb_len);
}
