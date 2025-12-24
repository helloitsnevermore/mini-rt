#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

// ============================================
// НАСТРОЙКИ ЭКСПЕРИМЕНТОВ
// ============================================
// Размер блока нитей (BLOCK_SIZE x BLOCK_SIZE). 
// Для эксперимента меняйте это значение: 16 или 32
#define BLOCK_SIZE 32 

// Тип данных (float для скорости, double для точности)
typedef float BaseType;

// ============================================
// МАКРОСЫ ДЛЯ ПРОВЕРКИ ОШИБОК
// ============================================
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error: %d at %s:%d\n", status, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================
// 1. НАИВНОЕ УМНОЖЕНИЕ (Global Memory)
// ============================================
__global__ void naiveMatrixMul(const BaseType *A, const BaseType *B, BaseType *C, int N) {
    // Глобальные индексы нити
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        BaseType sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ============================================
// 2. УМНОЖЕНИЕ С РАЗДЕЛЯЕМОЙ ПАМЯТЬЮ (Tiled Shared Memory)
// ============================================
__global__ void sharedMatrixMul(const BaseType *A, const BaseType *B, BaseType *C, int N) {
    // Статическое выделение разделяемой памяти
    __shared__ BaseType sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ BaseType sB[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    BaseType sum = 0.0f;

    // Цикл по тайлам (плиткам)
    // Мы двигаем плитку размером BLOCK_SIZE вдоль строки A и столбца B
    for (int m = 0; m < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {
        
        // Загрузка данных из глобальной памяти в разделяемую
        // Проверка границ обязательна, если N не кратно BLOCK_SIZE
        if (row < N && (m * BLOCK_SIZE + tx) < N)
            sA[ty][tx] = A[row * N + (m * BLOCK_SIZE + tx)];
        else
            sA[ty][tx] = 0.0f;

        if (col < N && (m * BLOCK_SIZE + ty) < N)
            sB[ty][tx] = B[(m * BLOCK_SIZE + ty) * N + col];
        else
            sB[ty][tx] = 0.0f;

        // Барьер: ждем, пока все нити блока загрузят свои данные
        __syncthreads();

        // Вычисление частичной суммы для текущего тайла
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += sA[ty][k] * sB[k][tx];
        }

        // Барьер: ждем, пока все нити закончат использовать текущие данные в sA и sB
        // перед тем, как на следующей итерации мы их перезапишем
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================
// ФУНКЦИИ-ОБЕРТКИ (HOST)
// ============================================

// Функция для запуска Наивного ядра
float runNaive(BaseType *d_A, BaseType *d_B, BaseType *d_C, int N) {
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    naiveMatrixMul<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    return milliseconds;
}

// Функция для запуска ядра с Shared Memory
float runShared(BaseType *d_A, BaseType *d_B, BaseType *d_C, int N) {
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    sharedMatrixMul<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    return milliseconds;
}

// Функция для запуска cuBLAS
float runCuBLAS(BaseType *d_A, BaseType *d_B, BaseType *d_C, int N) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    BaseType alpha = 1.0f;
    BaseType beta = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    // Внимание: cuBLAS использует column-major порядок.
    // C = alpha * A * B + beta * C
    // Чтобы получить корректный результат для row-major матриц C/C++,
    // мы вычисляем C^T = B^T * A^T.
    // Фактически мы меняем местами A и B в вызове функции.
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                N, N, N, 
                &alpha, 
                d_B, N, // B передаем как A
                d_A, N, // A передаем как B
                &beta, 
                d_C, N);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cublasDestroy(handle);
    return milliseconds;
}

// Проверка на CPU
void cpuMatrixMul(const BaseType *A, const BaseType *B, BaseType *C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            BaseType sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Сравнение результатов
bool verify(const BaseType *C_cpu, const BaseType *C_gpu, int N) {
    for (int i = 0; i < N * N; ++i) {
        if (fabs(C_cpu[i] - C_gpu[i]) > 1e-2) { // Допуск чуть больше для float
            printf("Mismatch at %d: CPU %f, GPU %f\n", i, C_cpu[i], C_gpu[i]);
            return false;
        }
    }
    return true;
}

int main() {
    // Список размеров матриц для тестирования
    // Убрал слишком маленькие, добавил побольше для нагрузки
    int sizes[] = {256, 512, 1024, 2048}; 
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("================================================================================\n");
    printf("Matrix Multiplication Benchmark (Block Size: %d)\n", BLOCK_SIZE);
    printf("================================================================================\n");
    printf("| Size | CPU (ms) | Naive (ms) | Shared (ms) | cuBLAS (ms) | Speedup (vs Naive) |\n");
    printf("|------|----------|------------|-------------|-------------|--------------------|\n");

    for (int i = 0; i < num_sizes; ++i) {
        int N = sizes[i];
        size_t bytes = N * N * sizeof(BaseType);

        // Выделение памяти на хосте
        BaseType *h_A = (BaseType*)malloc(bytes);
        BaseType *h_B = (BaseType*)malloc(bytes);
        BaseType *h_C_cpu = (BaseType*)malloc(bytes);
        BaseType *h_C_gpu = (BaseType*)malloc(bytes);

        // Инициализация
        for (int j = 0; j < N * N; ++j) {
            h_A[j] = 1.0f; 
            h_B[j] = 2.0f; // Результат в каждой ячейке должен быть 2.0 * N
        }

        // Выделение памяти на устройстве
        BaseType *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, bytes));
        CUDA_CHECK(cudaMalloc(&d_B, bytes));
        CUDA_CHECK(cudaMalloc(&d_C, bytes));

        // Копирование данных
        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

        // 1. CPU (только для малых размеров, иначе вечность)
        float cpu_time = 0.0f;
        if (N <= 1024) { 
            clock_t start = clock();
            cpuMatrixMul(h_A, h_B, h_C_cpu, N);
            clock_t end = clock();
            cpu_time = (float)(end - start) * 1000.0f / CLOCKS_PER_SEC;
        } else {
            cpu_time = -1.0f; // Skip
        }

        // 2. Naive GPU
        float naive_time = runNaive(d_A, d_B, d_C, N);
        
        // Проверка корректности (только для малых)
        if (N <= 512) {
            CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));
            if(!verify(h_C_cpu, h_C_gpu, N)) printf("Naive Check Failed!\n");
        }

        // 3. Shared Memory GPU
        float shared_time = runShared(d_A, d_B, d_C, N);
        
        if (N <= 512) {
            CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));
            if(!verify(h_C_cpu, h_C_gpu, N)) printf("Shared Check Failed!\n");
        }

        // 4. cuBLAS
        float cublas_time = runCuBLAS(d_A, d_B, d_C, N);

        // Вывод строки таблицы
        if (cpu_time > 0)
            printf("| %4d | %8.1f | %10.2f | %11.2f | %11.2f | %16.2fx |\n", 
                N, cpu_time, naive_time, shared_time, cublas_time, naive_time/shared_time);
        else
            printf("| %4d | %8s | %10.2f | %11.2f | %11.2f | %16.2fx |\n", 
                N, "-", naive_time, shared_time, cublas_time, naive_time/shared_time);

        // Очистка
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
    }

    return 0;
}