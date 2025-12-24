import matplotlib.pyplot as plt

procs = [1, 2, 4, 8, 16]
times = [6.408, 3.799, 3.673, 3.018, 3.156]

t1 = times[0]
speedup = [t1 / t for t in times]
efficiency = [s / p for s, p in zip(speedup, procs)]

plt.style.use('bmh')
plt.figure(figsize=(10, 6))
plt.plot(procs, times, marker='o', linestyle='-', linewidth=2, color='#1f77b4', label='Время работы')
plt.title('Зависимость времени выполнения от числа процессов', fontsize=14)
plt.xlabel('Число процессов (MPI)', fontsize=12)
plt.ylabel('Время (сек)', fontsize=12)
plt.xticks(procs)
plt.grid(True)
plt.legend()
plt.savefig('time_plot.png', dpi=300)
print("Сохранен: time_plot.png")
plt.close()


plt.figure(figsize=(10, 6))
plt.plot(procs, speedup, marker='o', linestyle='-', linewidth=2, color='#d62728', label='Реальное ускорение')
plt.title('График ускорения (Speedup)', fontsize=14)
plt.xlabel('Число процессов (MPI)', fontsize=12)
plt.ylabel('Ускорение (раз)', fontsize=12)
plt.xticks(procs)
plt.yticks(range(0, 18, 2))
plt.grid(True)
plt.legend()
plt.savefig('speedup_plot.png', dpi=300)
print("Сохранен: speedup_plot.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(procs, efficiency, marker='o', linestyle='-', linewidth=2, color='#2ca02c', label='Эффективность')

plt.title('График эффективности распараллеливания', fontsize=14)
plt.xlabel('Число процессов (MPI)', fontsize=12)
plt.ylabel('Эффективность (0.0 - 1.0)', fontsize=12)
plt.xticks(procs)
plt.ylim(0, 1.1)
plt.grid(True)
plt.legend()
plt.savefig('efficiency_plot.png', dpi=300)
print("Сохранен: efficiency_plot.png")
plt.close()