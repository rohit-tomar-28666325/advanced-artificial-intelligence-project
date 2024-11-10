import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

X_train = np.array([
    [0.630812147, 0.111274637],[1.481231638, 0.550879774],
    [0.085389312, -1.004405455],[1.185220936, 0.397592252],
    [0.44031619, 0.339426538],[-0.824500559, -0.207795318],
    [-0.326158756, -0.538912165],[1.629093681, -1.615757864],
    [0.88984675, -0.1831107],[0.551957891, -0.642928154]
])
X_test = np.array([[-2.63426953, -0.327913873],[0.796193449, -2.289942402]])
gamma = 0.5

def rbf_kernel_vector(x_test, X_train, gamma):
    kernel_vector = np.array([
        np.exp(-gamma * np.sum((x_test - x_train) ** 2))
        for x_train in X_train
    ])
    return kernel_vector

kernel_vectors = []
for x_test in X_test:
    kernel_vector = rbf_kernel_vector(x_test, X_train, gamma)
    print(f"RBF kernel vector for {x_test}:", kernel_vector)
    kernel_vectors.append(kernel_vector)

diff = np.abs(kernel_vector[0]-kernel_vector[1])
print("gamma=%s diff=%s" % (gamma, diff))

plt.figure(figsize=(8, 6))
sns.heatmap(kernel_vectors, annot=True, cmap='viridis', square=True, cbar=True)
plt.title("RBF Kernel Vectors")
plt.xlabel("X_train")
plt.ylabel("X_test")
plt.show()

