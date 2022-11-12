from numba import jit
import numpy as np
import time

def measure(x, steps):
    x() # to initiate jit compilation.
    start = time.perf_counter()
    for i in range(steps):
        x()

    end = time.perf_counter()
    return end - start

x = np.random.normal(size=(100, 100))
steps = 200

@jit(nopython=True)
def with_jit():
    trace = 0.0
    for i in range(x.shape[0]):
        trace += np.tanh(x[i, i])

def normal():
    trace = 0.0
    for i in range(x.shape[0]):
        trace += np.tanh(x[i, i])    

print("CPU Jit: {} secs".format(measure(with_jit, steps)))
print("CPU Normal: {} secs".format(measure(normal, steps)))
