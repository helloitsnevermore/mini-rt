#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdio>

// параметры сетки
const int NX = 384; 
const int NY = 384; 
const int NZ = 384;

// параметры области и уравнения
const double DX = 2.0, DY = 2.0, DZ = 2.0;
const double X_START = -1.0, Y_START = -1.0, Z_START = -1.0;
const double ALPHA = 1.0e5; // параметр уравнения
const double EPSILON = 1.0e-8; // порог сходимости

// точное решение
double get_phi_exact(double x, double y, double z) {
    return x * x + y * y + z * z;
}

// правая часть
double get_rho(double x, double y, double z) {
    return 6.0 - ALPHA * get_phi_exact(x, y, z);
}

// вспомогательная функция для индексов
// i - локальный индекс слоя (0..local_nx+1), j - Y, k - Z
inline int get_idx(int i, int j, int k) {
    return (i * NY + j) * NZ + k;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // шаги сетки
    double hx = DX / (NX - 1);
    double hy = DY / (NY - 1);
    double hz = DZ / (NZ - 1);

    // коэффициенты для метода Якоби
    double denom = 2.0/(hx*hx) + 2.0/(hy*hy) + 2.0/(hz*hz) + ALPHA;
    double cx = 1.0 / (hx*hx);
    double cy = 1.0 / (hy*hy);
    double cz = 1.0 / (hz*hz);

    // декомпозиция области
    int layers_per_proc = NX / world_size;
    int remainder = NX % world_size;
    
    // балансировка нагрузки, если NX не делится нацело
    int local_nx = layers_per_proc + (world_rank < remainder ? 1 : 0);
    
    // вычисляем глобальный индекс начала для текущего процесса
    int global_start_i = 0;
    for (int r = 0; r < world_rank; ++r) {
        global_start_i += layers_per_proc + (r < remainder ? 1 : 0);
    }

    // выделяем память: local_nx + 2 теневых слоя
    // слой 0 - теневой (данные от rank-1), слой local_nx+1 - теневой (данные от rank+1)
    // реальные данные: слои 1 .. local_nx
    int total_layers = local_nx + 2;
    int layer_size = NY * NZ;
    size_t total_size = total_layers * layer_size;

    double* phi_current = new double[total_size];
    double* phi_next    = new double[total_size];
    double* rho_arr     = new double[total_size];

    // инициализация и граничные условия
    for (int i = 0; i < total_layers; i++) {
        // перевод локального индекса i (где 1 - это первый реальный слой) в глобальный
        int global_i = global_start_i + (i - 1); 

        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                double x = X_START + global_i * hx;
                double y = Y_START + j * hy;
                double z = Z_START + k * hz;
                int idx = get_idx(i, j, k);

                rho_arr[idx] = get_rho(x, y, z);

                // если это глобальная граница области или начальное приближение
                if (global_i <= 0 || global_i >= NX - 1 || j == 0 || j == NY - 1 || k == 0 || k == NZ - 1) {
                    phi_current[idx] = get_phi_exact(x, y, z);
                } else {
                    phi_current[idx] = 0.0; // начальное приближение внутри - нули
                }
                phi_next[idx] = phi_current[idx];
            }
        }
    }

    int iter = 0;
    double max_diff_global = 1.0;
    
    MPI_Barrier(MPI_COMM_WORLD); // синхронизация перед замером времени
    double start_time = MPI_Wtime();

    while (max_diff_global > EPSILON) {
        double max_diff_local = 0.0;
        
        // асинхронный обмен границами
        MPI_Request reqs[4];
        int req_count = 0;

        // Ооптравка левой границы влево + прием левой тени оттуда
        if (world_rank > 0) {
            MPI_Irecv(&phi_current[get_idx(0, 0, 0)], layer_size, MPI_DOUBLE, world_rank - 1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
            MPI_Isend(&phi_current[get_idx(1, 0, 0)], layer_size, MPI_DOUBLE, world_rank - 1, 1, MPI_COMM_WORLD, &reqs[req_count++]);
        }
        // отправка правой границы вправо + прием правой тени оттуда
        if (world_rank < world_size - 1) {
            MPI_Irecv(&phi_current[get_idx(local_nx + 1, 0, 0)], layer_size, MPI_DOUBLE, world_rank + 1, 1, MPI_COMM_WORLD, &reqs[req_count++]);
            MPI_Isend(&phi_current[get_idx(local_nx, 0, 0)],     layer_size, MPI_DOUBLE, world_rank + 1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
        }

        // считаем внутреннюю часть - слои от 2 до local_nx-1
        for (int i = 2; i < local_nx; i++) {
            for (int j = 1; j < NY - 1; j++) {
                for (int k = 1; k < NZ - 1; k++) {
                    int idx = get_idx(i, j, k);
                    
                    // аппроксимация лапласиана (центр + соседи)
                    double laplace_sum = 
                        (phi_current[get_idx(i+1, j, k)] + phi_current[get_idx(i-1, j, k)]) * cx +
                        (phi_current[get_idx(i, j+1, k)] + phi_current[get_idx(i, j-1, k)]) * cy +
                        (phi_current[get_idx(i, j, k+1)] + phi_current[get_idx(i, j, k-1)]) * cz;
                    
                    phi_next[idx] = (laplace_sum - rho_arr[idx]) / denom;
                    
                    double diff = std::abs(phi_next[idx] - phi_current[idx]);
                    if (diff > max_diff_local) max_diff_local = diff;
                }
            }
        }

        // ожидаем завершение обменов
        if (req_count > 0) {
            MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
        }

        // считаем краевые слои (зависят от теневых)
        int borders[] = {1, local_nx};
        for (int b_idx = 0; b_idx < 2; b_idx++) {
            int i = borders[b_idx];
            
            bool is_global_boundary_left = (world_rank == 0 && i == 1);
            bool is_global_boundary_right = (world_rank == world_size - 1 && i == local_nx);
            
            if (is_global_boundary_left || is_global_boundary_right) continue;
            if (i >= local_nx + 1 || i <= 0) continue; 

            for (int j = 1; j < NY - 1; j++) {
                for (int k = 1; k < NZ - 1; k++) {
                    int idx = get_idx(i, j, k);
                    
                    double laplace_sum = 
                        (phi_current[get_idx(i+1, j, k)] + phi_current[get_idx(i-1, j, k)]) * cx +
                        (phi_current[get_idx(i, j+1, k)] + phi_current[get_idx(i, j-1, k)]) * cy +
                        (phi_current[get_idx(i, j, k+1)] + phi_current[get_idx(i, j, k-1)]) * cz;
                    
                    phi_next[idx] = (laplace_sum - rho_arr[idx]) / denom;
                    
                    double diff = std::abs(phi_next[idx] - phi_current[idx]);
                    if (diff > max_diff_local) max_diff_local = diff;
                }
            }
        }

        // свапаем указатели
        std::swap(phi_current, phi_next);

        // сбор глобальной ошибки
        MPI_Allreduce(&max_diff_local, &max_diff_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        iter++;
    }

    double end_time = MPI_Wtime();
    double execution_time = end_time - start_time;
    double max_execution_time;
    MPI_Reduce(&execution_time, &max_execution_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // проверка погрешности относительно точного решения
    double max_err_local = 0.0;
    for (int i = 1; i <= local_nx; i++) {
        int global_i = global_start_i + (i - 1);
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                double x = X_START + global_i * hx;
                double y = Y_START + j * hy;
                double z = Z_START + k * hz;
                int idx = get_idx(i, j, k);
                
                double err = std::abs(phi_current[idx] - get_phi_exact(x, y, z));
                if (err > max_err_local) max_err_local = err;
            }
        }
    }
    double max_err_global = 0.0;
    MPI_Reduce(&max_err_local, &max_err_global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // вывод
    if (world_rank == 0) {
        printf("MPI Lab 2: Jacobi Method 3D\n");
        printf("Grid: %d x %d x %d\n", NX, NY, NZ);
        printf("Processes: %d\n", world_size);
        printf("Iterations: %d\n", iter);
        printf("Time: %.6f s\n", max_execution_time);
        printf("Max Error: %e\n", max_err_global);
        printf("Result Delta: %e\n", max_diff_global);
    }

    delete[] phi_current;
    delete[] phi_next;
    delete[] rho_arr;

    MPI_Finalize();
    return 0;
}