/**
 * CoreX 跨 buffer 双精度减法 Bug 诊断程序
 *
 * 复现条件：在 GPU kernel 中，从两个不同的 cudaMalloc 分配的 device buffer
 * 读取 double 值并执行减法，结果可能完全错误。
 *
 * 用法：
 *   编译后直接运行，无需参数。程序会输出每项测试的 PASS/FAIL 状态。
 *   退出码 0 = 全部通过，非 0 = 存在失败。
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                      \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ============================================================================
// 测试 kernel
// ============================================================================

// 测试 1：跨 buffer 减法（最小复现）
//   out[0] = A[idx] (读 A)
//   out[1] = B[idx] (读 B)
//   out[2] = A[idx] - B[idx] (跨 buffer 减法)
__global__ void kernel_cross_buffer_sub(const double* A, const double* B,
                                        double* out, int idx)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    double a = A[idx];
    double b = B[idx];
    out[0] = a;
    out[1] = b;
    out[2] = a - b;
}

// 测试 2：同 buffer 减法（对照组）
//   out[0] = A[i], out[1] = A[j], out[2] = A[i] - A[j]
__global__ void kernel_same_buffer_sub(const double* A, double* out,
                                       int i, int j)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    double a = A[i];
    double b = A[j];
    out[0] = a;
    out[1] = b;
    out[2] = a - b;
}

// 测试 3：volatile 修饰（验证是否能绕过）
__global__ void kernel_cross_buffer_sub_volatile(const volatile double* A,
                                                  const volatile double* B,
                                                  double* out, int idx)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    double a = A[idx];
    double b = B[idx];
    out[0] = a;
    out[1] = b;
    out[2] = a - b;
}

// 测试 4：先拷到局部变量再减（验证寄存器行为）
__global__ void kernel_cross_buffer_sub_local_copy(const double* A,
                                                    const double* B,
                                                    double* out, int idx)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    double local_a = A[idx];
    double local_b = B[idx];
    __threadfence();  // 强制内存栅栏
    out[0] = local_a;
    out[1] = local_b;
    out[2] = local_a - local_b;
}

// 测试 5：用 __longlong_as_double 绕过（位操作验证）
__global__ void kernel_cross_buffer_sub_bitcast(const double* A,
                                                 const double* B,
                                                 double* out, int idx)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    // 以整数方式读取
    unsigned long long a_bits = *reinterpret_cast<const unsigned long long*>(&A[idx]);
    unsigned long long b_bits = *reinterpret_cast<const unsigned long long*>(&B[idx]);
    double a = __longlong_as_double(a_bits);
    double b = __longlong_as_double(b_bits);
    out[0] = a;
    out[1] = b;
    out[2] = a - b;
}

// 测试 6：多线程并行跨 buffer 减法
__global__ void kernel_cross_buffer_sub_parallel(int n, const double* A,
                                                  const double* B,
                                                  double* diff)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    diff[i] = A[i] - B[i];
}

// 测试 7：float 单精度跨 buffer 减法（对比 double）
__global__ void kernel_cross_buffer_sub_float(const float* A, const float* B,
                                               float* out, int idx)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float a = A[idx];
    float b = B[idx];
    out[0] = a;
    out[1] = b;
    out[2] = a - b;
}

// 测试 8：跨 buffer 加法（验证是否仅减法有问题）
__global__ void kernel_cross_buffer_add(const double* A, const double* B,
                                         double* out, int idx)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    double a = A[idx];
    double b = B[idx];
    out[0] = a;
    out[1] = b;
    out[2] = a + b;
}

// 测试 9：跨 buffer 乘法
__global__ void kernel_cross_buffer_mul(const double* A, const double* B,
                                         double* out, int idx)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    double a = A[idx];
    double b = B[idx];
    out[0] = a;
    out[1] = b;
    out[2] = a * b;
}

// ============================================================================
// 辅助函数
// ============================================================================

static int g_pass = 0, g_fail = 0;

void report(const char* name, bool pass, const char* detail = nullptr)
{
    if (pass) {
        printf("  [PASS] %s", name);
        g_pass++;
    } else {
        printf("  [FAIL] %s", name);
        g_fail++;
    }
    if (detail) printf("  — %s", detail);
    printf("\n");
}

// ============================================================================
// 测试用例
// ============================================================================

void test_cross_buffer_subtract_zero()
{
    printf("\n=== 测试 1: 跨 buffer 减法 (0.0 - 0.0) ===\n");

    double h_val = 0.0;
    double *d_A, *d_B, *d_out;
    double h_out[3];

    CHECK_CUDA(cudaMalloc(&d_A, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_B, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_out, 3 * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_A, &h_val, sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, &h_val, sizeof(double), cudaMemcpyHostToDevice));

    kernel_cross_buffer_sub<<<1, 1>>>(d_A, d_B, d_out, 0);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_out, d_out, 3 * sizeof(double), cudaMemcpyDeviceToHost));

    char buf[256];
    snprintf(buf, sizeof(buf), "read_A=%.17g, read_B=%.17g, A-B=%.17g (期望 0.0)",
             h_out[0], h_out[1], h_out[2]);
    report("0.0 - 0.0 跨 buffer", h_out[2] == 0.0, buf);

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_out));
}

void test_cross_buffer_subtract_nonzero()
{
    printf("\n=== 测试 2: 跨 buffer 减法 (非零值) ===\n");

    double h_A = 3.14159265358979;
    double h_B = 1.41421356237310;
    double expected = h_A - h_B;
    double *d_A, *d_B, *d_out;
    double h_out[3];

    CHECK_CUDA(cudaMalloc(&d_A, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_B, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_out, 3 * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_A, &h_A, sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, &h_B, sizeof(double), cudaMemcpyHostToDevice));

    kernel_cross_buffer_sub<<<1, 1>>>(d_A, d_B, d_out, 0);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_out, d_out, 3 * sizeof(double), cudaMemcpyDeviceToHost));

    double err = fabs(h_out[2] - expected);
    char buf[256];
    snprintf(buf, sizeof(buf),
             "read_A=%.17g, read_B=%.17g, A-B=%.17g, 期望=%.17g, 误差=%.2e",
             h_out[0], h_out[1], h_out[2], expected, err);
    report("3.14... - 1.41... 跨 buffer", err < 1e-15, buf);

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_out));
}

void test_same_buffer_subtract()
{
    printf("\n=== 测试 3: 同 buffer 减法 (对照组) ===\n");

    double h_vals[2] = {0.0, 0.0};
    double *d_A, *d_out;
    double h_out[3];

    CHECK_CUDA(cudaMalloc(&d_A, 2 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_out, 3 * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_A, h_vals, 2 * sizeof(double), cudaMemcpyHostToDevice));

    kernel_same_buffer_sub<<<1, 1>>>(d_A, d_out, 0, 1);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_out, d_out, 3 * sizeof(double), cudaMemcpyDeviceToHost));

    char buf[256];
    snprintf(buf, sizeof(buf), "read_A[0]=%.17g, read_A[1]=%.17g, A[0]-A[1]=%.17g",
             h_out[0], h_out[1], h_out[2]);
    report("0.0 - 0.0 同 buffer", h_out[2] == 0.0, buf);

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_out));
}

void test_volatile_workaround()
{
    printf("\n=== 测试 4: volatile 修饰 (尝试绕过) ===\n");

    double h_val = 0.0;
    double *d_A, *d_B, *d_out;
    double h_out[3];

    CHECK_CUDA(cudaMalloc(&d_A, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_B, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_out, 3 * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_A, &h_val, sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, &h_val, sizeof(double), cudaMemcpyHostToDevice));

    kernel_cross_buffer_sub_volatile<<<1, 1>>>(d_A, d_B, d_out, 0);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_out, d_out, 3 * sizeof(double), cudaMemcpyDeviceToHost));

    char buf[256];
    snprintf(buf, sizeof(buf), "A-B=%.17g (期望 0.0)", h_out[2]);
    report("volatile 跨 buffer 0.0-0.0", h_out[2] == 0.0, buf);

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_out));
}

void test_threadfence_workaround()
{
    printf("\n=== 测试 5: __threadfence + 局部变量 (尝试绕过) ===\n");

    double h_val = 0.0;
    double *d_A, *d_B, *d_out;
    double h_out[3];

    CHECK_CUDA(cudaMalloc(&d_A, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_B, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_out, 3 * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_A, &h_val, sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, &h_val, sizeof(double), cudaMemcpyHostToDevice));

    kernel_cross_buffer_sub_local_copy<<<1, 1>>>(d_A, d_B, d_out, 0);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_out, d_out, 3 * sizeof(double), cudaMemcpyDeviceToHost));

    char buf[256];
    snprintf(buf, sizeof(buf), "A-B=%.17g (期望 0.0)", h_out[2]);
    report("__threadfence 跨 buffer 0.0-0.0", h_out[2] == 0.0, buf);

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_out));
}

void test_bitcast_workaround()
{
    printf("\n=== 测试 6: 位操作读取 + 减法 (尝试绕过) ===\n");

    double h_val = 0.0;
    double *d_A, *d_B, *d_out;
    double h_out[3];

    CHECK_CUDA(cudaMalloc(&d_A, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_B, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_out, 3 * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_A, &h_val, sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, &h_val, sizeof(double), cudaMemcpyHostToDevice));

    kernel_cross_buffer_sub_bitcast<<<1, 1>>>(d_A, d_B, d_out, 0);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_out, d_out, 3 * sizeof(double), cudaMemcpyDeviceToHost));

    char buf[256];
    snprintf(buf, sizeof(buf), "A-B=%.17g (期望 0.0)", h_out[2]);
    report("bitcast 跨 buffer 0.0-0.0", h_out[2] == 0.0, buf);

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_out));
}

void test_parallel_cross_buffer()
{
    printf("\n=== 测试 7: 多线程并行跨 buffer 减法 (N=1024) ===\n");

    const int N = 1024;
    std::vector<double> h_A(N), h_B(N);
    for (int i = 0; i < N; i++) {
        h_A[i] = (double)i * 0.001;
        h_B[i] = (double)i * 0.001;  // 相同值，期望差为 0
    }

    double *d_A, *d_B, *d_diff;
    CHECK_CUDA(cudaMalloc(&d_A, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_B, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_diff, N * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    kernel_cross_buffer_sub_parallel<<<(N + 255) / 256, 256>>>(N, d_A, d_B, d_diff);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<double> h_diff(N);
    CHECK_CUDA(cudaMemcpy(h_diff.data(), d_diff, N * sizeof(double), cudaMemcpyDeviceToHost));

    int err_count = 0;
    double max_err = 0.0;
    int worst_idx = -1;
    for (int i = 0; i < N; i++) {
        if (h_diff[i] != 0.0) {
            err_count++;
            if (fabs(h_diff[i]) > max_err) {
                max_err = fabs(h_diff[i]);
                worst_idx = i;
            }
        }
    }

    char buf[256];
    if (err_count > 0) {
        snprintf(buf, sizeof(buf),
                 "%d/%d 元素错误, 最大误差=%.17g (idx=%d, A=%.17g, B=%.17g, diff=%.17g)",
                 err_count, N, max_err, worst_idx,
                 h_A[worst_idx], h_B[worst_idx], h_diff[worst_idx]);
    } else {
        snprintf(buf, sizeof(buf), "全部 %d 元素差值为 0.0", N);
    }
    report("并行跨 buffer (相同值)", err_count == 0, buf);

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_diff));
}

void test_float_cross_buffer()
{
    printf("\n=== 测试 8: 单精度 float 跨 buffer 减法 (对比 double) ===\n");

    float h_val = 0.0f;
    float *d_A, *d_B, *d_out;
    float h_out[3];

    CHECK_CUDA(cudaMalloc(&d_A, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, 3 * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, &h_val, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, &h_val, sizeof(float), cudaMemcpyHostToDevice));

    kernel_cross_buffer_sub_float<<<1, 1>>>(d_A, d_B, d_out, 0);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_out, d_out, 3 * sizeof(float), cudaMemcpyDeviceToHost));

    char buf[256];
    snprintf(buf, sizeof(buf), "A-B=%.10g (期望 0.0)", (double)h_out[2]);
    report("float 跨 buffer 0.0-0.0", h_out[2] == 0.0f, buf);

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_out));
}

void test_cross_buffer_add()
{
    printf("\n=== 测试 9: 跨 buffer 加法 (0.0 + 0.0) ===\n");

    double h_val = 0.0;
    double *d_A, *d_B, *d_out;
    double h_out[3];

    CHECK_CUDA(cudaMalloc(&d_A, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_B, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_out, 3 * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_A, &h_val, sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, &h_val, sizeof(double), cudaMemcpyHostToDevice));

    kernel_cross_buffer_add<<<1, 1>>>(d_A, d_B, d_out, 0);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_out, d_out, 3 * sizeof(double), cudaMemcpyDeviceToHost));

    char buf[256];
    snprintf(buf, sizeof(buf), "A+B=%.17g (期望 0.0)", h_out[2]);
    report("double 跨 buffer 加法 0.0+0.0", h_out[2] == 0.0, buf);

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_out));
}

void test_cross_buffer_add_nonzero()
{
    printf("\n=== 测试 10: 跨 buffer 加法 (非零) ===\n");

    double h_A = 1.5, h_B = 2.5;
    double expected = h_A + h_B;
    double *d_A, *d_B, *d_out;
    double h_out[3];

    CHECK_CUDA(cudaMalloc(&d_A, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_B, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_out, 3 * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_A, &h_A, sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, &h_B, sizeof(double), cudaMemcpyHostToDevice));

    kernel_cross_buffer_add<<<1, 1>>>(d_A, d_B, d_out, 0);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_out, d_out, 3 * sizeof(double), cudaMemcpyDeviceToHost));

    double err = fabs(h_out[2] - expected);
    char buf[256];
    snprintf(buf, sizeof(buf), "A+B=%.17g, 期望=%.17g, 误差=%.2e",
             h_out[2], expected, err);
    report("double 跨 buffer 加法 1.5+2.5", err < 1e-15, buf);

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_out));
}

void test_cross_buffer_mul_nonzero()
{
    printf("\n=== 测试 11: 跨 buffer 乘法 (非零) ===\n");

    double h_A = 3.0, h_B = 7.0;
    double expected = h_A * h_B;
    double *d_A, *d_B, *d_out;
    double h_out[3];

    CHECK_CUDA(cudaMalloc(&d_A, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_B, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_out, 3 * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_A, &h_A, sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, &h_B, sizeof(double), cudaMemcpyHostToDevice));

    kernel_cross_buffer_mul<<<1, 1>>>(d_A, d_B, d_out, 0);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_out, d_out, 3 * sizeof(double), cudaMemcpyDeviceToHost));

    double err = fabs(h_out[2] - expected);
    char buf[256];
    snprintf(buf, sizeof(buf), "A*B=%.17g, 期望=%.17g, 误差=%.2e",
             h_out[2], expected, err);
    report("double 跨 buffer 乘法 3.0*7.0", err < 1e-15, buf);

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_out));
}

void test_large_buffer_cross_subtract()
{
    printf("\n=== 测试 12: 大 buffer 跨 buffer 减法 (模拟实际仿真场景) ===\n");
    printf("    模拟 libuipc 场景: 两个 12-double 向量 buffer (96 字节, 仿射体 DOF)\n");

    const int N = 12;
    std::vector<double> h_A(N), h_B(N);
    // 模拟 q 和 q_tilde 都是相同值的情况
    for (int i = 0; i < N; i++) {
        h_A[i] = (i == 0) ? 0.0 : (i == 1) ? 2.3 : 0.0;
        h_B[i] = h_A[i];  // 相同值
    }

    double *d_A, *d_B, *d_diff;
    CHECK_CUDA(cudaMalloc(&d_A, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_B, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_diff, N * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    kernel_cross_buffer_sub_parallel<<<1, N>>>(N, d_A, d_B, d_diff);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<double> h_diff(N);
    CHECK_CUDA(cudaMemcpy(h_diff.data(), d_diff, N * sizeof(double), cudaMemcpyDeviceToHost));

    bool all_ok = true;
    printf("    逐元素结果:\n");
    for (int i = 0; i < N; i++) {
        bool ok = (h_diff[i] == 0.0);
        if (!ok) all_ok = false;
        printf("      [%2d] A=%.17g  B=%.17g  A-B=%.17g  %s\n",
               i, h_A[i], h_B[i], h_diff[i], ok ? "OK" : "*** WRONG ***");
    }

    report("12-double 跨 buffer 减法", all_ok,
           all_ok ? "全部正确" : "存在跨 buffer 减法错误");

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_diff));
}

// ============================================================================
// 主函数
// ============================================================================

int main()
{
    printf("================================================================\n");
    printf(" CoreX 跨 buffer 双精度减法 Bug 诊断程序\n");
    printf("================================================================\n");

    // 打印设备信息
    int dev;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDevice(&dev));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    printf("设备: %s (compute %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("CUDA Runtime: %d\n\n", CUDART_VERSION);

    // 核心测试
    test_cross_buffer_subtract_zero();       // 1: 跨 buffer 0-0 (最小复现)
    test_cross_buffer_subtract_nonzero();    // 2: 跨 buffer 非零值
    test_same_buffer_subtract();             // 3: 同 buffer 0-0 (对照)

    // 绕过尝试
    test_volatile_workaround();              // 4: volatile
    test_threadfence_workaround();           // 5: __threadfence
    test_bitcast_workaround();               // 6: 位操作读取

    // 扩展测试
    test_parallel_cross_buffer();            // 7: 多线程并行
    test_float_cross_buffer();              // 8: float 单精度对比
    test_cross_buffer_add();                // 9: 加法 0+0
    test_cross_buffer_add_nonzero();        // 10: 加法非零
    test_cross_buffer_mul_nonzero();        // 11: 乘法非零

    // 仿真场景复现
    test_large_buffer_cross_subtract();     // 12: 模拟实际仿真 DOF buffer

    // 汇总
    printf("\n================================================================\n");
    printf(" 总计: %d PASS, %d FAIL\n", g_pass, g_fail);
    printf("================================================================\n");

    if (g_fail > 0) {
        printf("\n*** 检测到跨 buffer 双精度运算异常 ***\n");
        printf("影响: 在 GPU kernel 中从两个不同的 cudaMalloc buffer\n");
        printf("      读取 double 值并执行算术运算时，结果可能完全错误。\n");
        printf("      单独读取每个值正确，仅跨 buffer 运算结果异常。\n");
    }

    return g_fail > 0 ? 1 : 0;
}
