#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "common.h"
#include "mmio.h"
#include "mmio_highlevel.h"
#include "recblocking_solver.h"
#include "recblocking_solver_cuda.h"

#include "unisolver/ArrayUtils.hpp"
#include "unisolver/JsonUtils.hpp"

using namespace uni;

#define CHECK_CUDA(func)                                               \
    {                                                                  \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess) {                                   \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
            while (1);                                                 \
        }                                                              \
    }

#define CHECK_CUSPARSE(func)                                               \
    {                                                                      \
        cusparseStatus_t status = (func);                                  \
        if (status != CUSPARSE_STATUS_SUCCESS) {                           \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cusparseGetErrorString(status), status);      \
            while (1);                                                     \
        }                                                                  \
    }

using cusp_int = int;
#define my_CUSPARSE_INDEX CUSPARSE_INDEX_32I

#define MAX_DOF_TEST 8

struct benchmark_record {
    double total_time = 0;
    long flops = 0;
    long bytes = 0;
    long count = 0;
};

benchmark_record benchmark_record_map_lower[MAX_DOF_TEST];

double recblock_sptrsv_csr(int m, int n, int nnzTR, int *csrRowPtrTR,
                           int *csrColIdxTR, VALUE_TYPE *csrValTR,
                           VALUE_TYPE *b, VALUE_TYPE *x, int lv) {
    int device_id = 0;
    if (lv == -1) {
        int li = 1;
        for (li = 1; li <= 100; li++) {
            if (m / pow(2, (li + 1)) <
                (device_id == 0 ? 92160
                                : 58880))  // 92160 (4608x20) is titan rtx,
                                           // 58880 (2944x20) is rtx 2080
                break;
        }
        lv = li;
    }

    int rhs = 1;
    int substitution = SUBSTITUTION_FORWARD;

    // transpose CSR of U and L to CSC
    int *cscColPtrTR = (int *)malloc(sizeof(int) * (n + 1));
    cscColPtrTR[0] = 0;
    int *cscRowIdxTR = (int *)malloc(sizeof(int) * nnzTR);
    VALUE_TYPE *cscValTR = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * nnzTR);
    matrix_transposition(m, n, nnzTR, csrRowPtrTR, csrColIdxTR, csrValTR,
                         cscRowIdxTR, cscColPtrTR, cscValTR);

    if (lv == -1) {
        int li = 1;
        for (li = 1; li <= 100; li++) {
            if (m / pow(2, (li + 1)) <
                (device_id == 0 ? 92160
                                : 58880))  // 92160 (4608x20) is titan rtx,
                                           // 58880 (2944x20) is rtx 2080
                break;
        }
        lv = li;
    }

    int *d_cscColPtrTR;
    int *d_cscRowIdxTR;
    VALUE_TYPE *d_cscValTR;
    cudaMalloc((void **)&d_cscColPtrTR, (n + 1) * sizeof(int));
    cudaMalloc((void **)&d_cscRowIdxTR, nnzTR * sizeof(int));
    cudaMalloc((void **)&d_cscValTR, nnzTR * sizeof(VALUE_TYPE));

    cudaMemcpy(d_cscColPtrTR, cscColPtrTR, sizeof(int) * (n + 1),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_cscRowIdxTR, cscRowIdxTR, sizeof(int) * nnzTR,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_cscValTR, cscValTR, sizeof(VALUE_TYPE) * nnzTR,
               cudaMemcpyHostToDevice);

    VALUE_TYPE *d_x;
    VALUE_TYPE *d_b;
    cudaMalloc((void **)&d_x, m * sizeof(VALUE_TYPE));
    cudaMalloc((void **)&d_b, m * sizeof(VALUE_TYPE));

    cudaMemcpy(d_x, x, sizeof(VALUE_TYPE) * m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(VALUE_TYPE) * m, cudaMemcpyHostToDevice);

    double cal_time = 0;
    double preprocess_time = 0;
    recblocking_solver_cuda(d_cscColPtrTR, d_cscRowIdxTR, d_cscValTR, m, n,
                            nnzTR, d_x, d_b, substitution, lv, &cal_time,
                            &preprocess_time);
    cudaMemcpy(x, d_x, sizeof(VALUE_TYPE) * m, cudaMemcpyDeviceToHost);

    printf("Preprocess time = %.3lf ms\n", preprocess_time);
    printf("computation usetime = %.3lf ms\n", cal_time);
    printf("Performance = %.3lf gflops\n", (2 * nnzTR) / (cal_time * 1e6));

    cudaFree(d_cscColPtrTR);
    cudaFree(d_cscRowIdxTR);
    cudaFree(d_cscValTR);
    cudaFree(d_b);
    cudaFree(d_x);
    free(cscColPtrTR);
    free(cscRowIdxTR);
    free(cscValTR);

    return cal_time;
}

void RunBenchmarkLowerWithCusparse(int Dof, int stencil_type, int stencil_width,
                                   int M, int N, int P, int lv) {
    constexpr int Dim = 3;

    std::vector<std::array<cusp_int, Dim>> stencil_points;
    if (stencil_type == 0) {
        for (int d = Dim - 1; d >= 0; d--) {
            for (int j = stencil_width; j > 0; j--) {
                std::array<cusp_int, Dim> pt = {0, 0, 0};
                pt[d] = -j;
                stencil_points.push_back(pt);
            }
        }
        stencil_points.push_back(std::array<cusp_int, Dim>{0, 0, 0});
    } else if (stencil_type == 1) {
        NestedLoop(
            constant_array<cusp_int, Dim>(-stencil_width),
            constant_array<cusp_int, Dim>(2 * stencil_width + 1), [&](auto pt) {
                cusp_int cnt = CartToFlat(
                    pt + stencil_width,
                    constant_array<cusp_int, Dim>(2 * stencil_width + 1));
                if (cnt < (myPow(2 * stencil_width + 1, Dim) / 2)) {
                    stencil_points.push_back(pt);
                }
            });
        stencil_points.push_back(std::array<cusp_int, Dim>{0, 0, 0});
    } else if (stencil_type == 2) {
        stencil_points.push_back(std::array<cusp_int, Dim>{0, 0, -1});
        stencil_points.push_back(std::array<cusp_int, Dim>{1, 0, -1});
        stencil_points.push_back(std::array<cusp_int, Dim>{0, 1, -1});
        stencil_points.push_back(std::array<cusp_int, Dim>{0, -1, 0});
        stencil_points.push_back(std::array<cusp_int, Dim>{1, -1, 0});
        stencil_points.push_back(std::array<cusp_int, Dim>{-1, 0, 0});
        stencil_points.push_back(std::array<cusp_int, Dim>{0, 0, 0});
    } else {
        stencil_points.push_back(std::array<cusp_int, Dim>{0, 0, -2});
        stencil_points.push_back(std::array<cusp_int, Dim>{0, -1, -1});
        stencil_points.push_back(std::array<cusp_int, Dim>{-1, 0, -1});
        stencil_points.push_back(std::array<cusp_int, Dim>{0, -2, 0});
        stencil_points.push_back(std::array<cusp_int, Dim>{0, 0, -1});
        stencil_points.push_back(std::array<cusp_int, Dim>{-1, -1, 0});
        stencil_points.push_back(std::array<cusp_int, Dim>{1, 0, -1});
        stencil_points.push_back(std::array<cusp_int, Dim>{0, -1, 0});
        stencil_points.push_back(std::array<cusp_int, Dim>{-2, 0, 0});
        stencil_points.push_back(std::array<cusp_int, Dim>{0, 1, -1});
        stencil_points.push_back(std::array<cusp_int, Dim>{1, -1, 0});
        stencil_points.push_back(std::array<cusp_int, Dim>{-1, 0, 0});
        stencil_points.push_back(std::array<cusp_int, Dim>{0, 0, 0});
    }

    // Host problem definition
    cusp_int A_num_rows = M * N * P * Dof;
    cusp_int A_nnz = 0;
    std::vector<cusp_int> hA_csrOffsets;
    std::vector<cusp_int> hA_columns;
    std::vector<double> hA_values;
    std::vector<double> hX;
    std::vector<double> hY;
    std::vector<double> hY_result;
    // 注意这里求解的是A* Y = X, 所以这里的Y是输出, X是输入

    // set A & hX
    NestedLoop(
        std::array<cusp_int, Dim>{}, std::array<cusp_int, Dim>{M, N, P},
        [&](auto loc) {
            for (int d = 0; d < Dof; d++) {
                hA_csrOffsets.push_back(A_nnz);
                cusp_int cnt = 0;
                for (auto pt : stencil_points) {
                    if (in_range(loc + pt, std::array<cusp_int, Dim>{},
                                 std::array<cusp_int, Dim>{M, N, P} - 1)) {
                        for (int k = 0; k < Dof; k++) {
                            if (pt != std::array<cusp_int, Dim>{0, 0, 0} ||
                                k == d) {
                                hA_columns.push_back(
                                    CartToFlat(
                                        loc + pt,
                                        std::array<cusp_int, Dim>{M, N, P}) *
                                        Dof +
                                    k);
                                hA_values.push_back(1.);
                                A_nnz++;
                                cnt++;
                            }
                        }
                    }
                }
                hX.push_back(cnt);
            }
        });
    hA_csrOffsets.push_back(A_nnz);

    std::cout << "A_nnz = " << A_nnz << "\n";

    // set hY
    hY.resize(A_num_rows);
    hY_result.resize(A_num_rows);
    for (cusp_int i = 0; i < A_num_rows; i++) hY_result[i] = 1.0;

    //--------------------------------------------------------------------------
    /* !!!!!! start computing SpTRSV !!!!!!!! */

    // warm up
    double solve_time = recblock_sptrsv_csr(
        A_num_rows, A_num_rows, A_nnz, hA_csrOffsets.data(), hA_columns.data(),
        hA_values.data(), hX.data(), hY.data(), lv);
    // test
    /////

    long readBytes = (sizeof(cusp_int) + sizeof(double)) * A_nnz +
                     sizeof(cusp_int) * A_num_rows +
                     sizeof(double) * A_num_rows;
    long writeBytes = sizeof(double) * A_num_rows;

    benchmark_record_map_lower[Dof - 1] = {solve_time * 1e-3 * 10,
                                           2L * A_nnz * 10,
                                           (readBytes + writeBytes) * 10, 10};

    //--------------------------------------------------------------------------
    // device result check

    int correct = 1;
    for (cusp_int i = 0; i < A_num_rows; i++) {
        if (hY[i] !=
            hY_result[i]) {  // direct doubleing point comparison is not
            correct = 0;     // reliable
            // break;
            std::cout << "i = " << i << ", hY[i] = " << hY[i]
                      << ", hY_result[i] = " << hY_result[i] << std::endl;
        }
    }
    if (correct)
        printf("recblock test PASSED\n");
    else
        printf("recblock test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // device memory deallocation
    // log::FunctionEnd(0, 0, 0);
}

int main(int argc, char **argv) {
    std::string problems[] = {"stencilstar", "stencilbox", "stencilstarfill1",
                              "stencildiamond"};

    assert(argc > 6);
    int i = atoi(argv[1]);
    int stencil_width_0 = atoi(argv[2]);
    int dof = atoi(argv[3]) - 1;
    cusp_int M = atoi(argv[4]);
    cusp_int N = atoi(argv[5]);
    cusp_int P = atoi(argv[6]);
    int lv = argc > 7 ? atoi(argv[7]) : -1;

    int stencil_width = stencil_width_0 + 1;
    std::string problem = problems[i];

    std::cout << problem << ", width=" << stencil_width << ", dof=" << dof + 1
              << std::endl;

    std::cout << "\tmesh size=" << M << 'x' << N << 'x' << P << std::endl;
    RunBenchmarkLowerWithCusparse(dof + 1, i, stencil_width, M, N, P, lv);
    std::cout << "\t\tLower:";
    double total_time = benchmark_record_map_lower[dof].total_time;
    double total_flops_time =
        static_cast<double>(benchmark_record_map_lower[dof].flops) / total_time;
    double total_bytes_time =
        static_cast<double>(benchmark_record_map_lower[dof].bytes) / total_time;

    std::cout << dof + 1 << "," << total_time << "," << total_flops_time * 1e-9
              << "," << total_bytes_time * 1e-9 << std::endl;

    return 0;
}
