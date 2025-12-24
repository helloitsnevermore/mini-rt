import matplotlib.pyplot as plt
import numpy as np

# Данные из твоего скриншота
sizes = [256, 512, 1024, 2048, 4096]
time_naive = [0.27, 1.17, 9.20, 74.54, 358.15]
time_shared = [0.13, 0.77, 5.81, 44.04, 170.42]
time_cublas = [10.34, 0.32, 0.92, 5.87, 21.57] # Первое значение - инициализация

# ==========================================
# График 1: Сравнение GPU методов (Линейный масштаб)
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(sizes, time_naive, marker='o', label='Naive (Global Mem)', linewidth=2)
plt.plot(sizes, time_shared, marker='s', label='Shared Memory (Tiled)', linewidth=2)
plt.plot(sizes, time_cublas, marker='^', label='cuBLAS', linewidth=2)

plt.title('Производительность умножения матриц на GPU', fontsize=14)
plt.xlabel('Размер матрицы (NxN)', fontsize=12)
plt.ylabel('Время выполнения (мс)', fontsize=12)
plt.xticks(sizes)
plt.grid(True)
plt.legend()
# Уберем аномалию первого запуска cuBLAS из масштаба графика, 
# чтобы было видно остальные точки
plt.ylim(0, 400) 

plt.savefig('gpu_comparison.png', dpi=300)
print("Saved: gpu_comparison.png")
plt.close()

# ==========================================
# График 2: Ускорение Shared относительно Naive
# ==========================================
speedup = [n / s for n, s in zip(time_naive, time_shared)]

plt.figure(figsize=(10, 6))
plt.plot(sizes, speedup, marker='o', color='green', linewidth=2)
plt.axhline(y=1.0, color='gray', linestyle='--')

plt.title('Ускорение Shared Memory относительно Naive', fontsize=14)
plt.xlabel('Размер матрицы (NxN)', fontsize=12)
plt.ylabel('Ускорение (раз)', fontsize=12)
plt.xticks(sizes)
plt.ylim(0, 3.0)
plt.grid(True)

for i, txt in enumerate(speedup):
    plt.annotate(f"{txt:.2f}x", (sizes[i], speedup[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.savefig('speedup_shared.png', dpi=300)
print("Saved: speedup_shared.png")
plt.close()