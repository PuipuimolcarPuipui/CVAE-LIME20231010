import matplotlib.pyplot as plt
import numpy as np

def combinations(N, k):
    from math import factorial
    return factorial(N) / (factorial(k) * factorial(N - k))

def data_count(N, k):
    return combinations(N, k) * (3 ** k)

N_max = 50
points = np.array(range(N_max + 1))  # Convert to NumPy array for element-wise operations

for N in range(1, N_max + 1, 5):
    data_counts = np.array([data_count(N, k) for k in range(N + 1)])  # Convert to NumPy array
    sum_counts = np.sum(data_counts)
    plt.plot(points[:N+1] / N, data_counts / sum_counts, label=f'N={N}')  # Element-wise division

plt.xlabel('Points')
plt.ylabel('Weight')
plt.legend()
plt.show()