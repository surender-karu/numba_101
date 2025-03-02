from numba import cuda, float32

TBP = 16

@cuda.jit
def fast_matmul(A, B, C):
    sA = cuda.shared.array(shpe=(TBP, TBP), dtype=float32)
    sB = cuda.shared.array(shpe=(TBP, TBP), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    bpg = cuda.gridDim.x

    if x >= C.shape[0] and y >= C.shape[1]:
        return

    tmp = 0.0

    for i in range(bpg):
        sA[tx, ty] = A[x, ty + i * TBP]
        sB[tx, ty] = B[x, ty + i * TBP]

        cuda.syncthreads()

        for j in range(TBP):
            tmp += sA[tx, j] * sB[j, ty]

        cuda.syncthreads()

        C[x, y] = tmp

