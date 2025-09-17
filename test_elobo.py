from elobo_project import run_least_squares, run_elobo_efficient
import numpy as np
import pandas as pd

# Test data

A = np.array([
    [ 1,  0 ],
    [ 1,  0 ],
    [-1,  1 ],
    [ 0, -1 ],
    [ 0, -1 ]
])

b = np.array([
    -246.23,
    -223.15,
    0,
    210.30,
    234.21
]).reshape(-1, 1)

y = np.array([
    -5.528,
    17.547,
    -14.270,
    -16.132,
    7.783
]).reshape(-1, 1)

Q = np.diag([1, 3, 2, 1, 2])

sigma2 = 4e-4
threshold = 0.5


blocks = [
    [0], [1], [2], [3], [4]
]

# Step 1: Run Least Squares
x_hat, y_fit, residuals, sigma2_hat = run_least_squares(A, b, y, sigma2, Q)

print("———— Step 1: Least Squares Results ————")
print("x_hat (H3, H4):", x_hat.flatten())
print("Residuals:", residuals.flatten())
print("Sigma²_hat:", sigma2_hat)


# Step 2: Run ELOBO
df = run_elobo_efficient(A, b, y, sigma2, blocks, threshold, Q)

print("———— Step 2: ELOBO Results ————")
print(df)
