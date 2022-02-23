import numpy as np
from numpy.linalg import matrix_power


def get_A(Tr):
    A = np.zeros((Tr + 2, Tr + 2))
    A[:-1, 1:] = np.identity(Tr + 1)
    A[0, 0] = 1.0
    A[-1, 0] = 1.0
    A[-1, 1] = 1.0

    return A


def card(N, L, Tr, cyclic=True):
    if N < 1:
        raise ValueError(f"N must be stricly positive")

    if L < 1:
        raise ValueError(f"L must be stricly positive")

    if Tr < 0:
        raise ValueError(f"Tr must be positive")

    A = get_A(Tr)

    if cyclic:
        return int(np.trace(matrix_power(A, L)) ** N)

    if L <= Tr:
        return int((L + 1) ** N)

    return int(np.sum(matrix_power(A, L - Tr - 1)) ** N)


# def f_set(L, Tr):

#     if L < 1:
#         raise ValueError(f"L must be stricly positive")

#     if Tr < 0:
#         raise ValueError(f"Tr must be positive")

#     if Tr == 0:
#         return b_set(L)

#     def recursion(L, Tr):
#         if L == 1:
#             return {(0,)}

#         if 2 <= L <= Tr:
#             return {seq + (0,) for seq in recursion(L - 1, Tr)}

#         if L == Tr + 1:
#             seq1 = {seq + (0,) for seq in recursion(L - 1, Tr)}
#             seq2 = {(0,) * k + (1,) + (0,) * (Tr - 1 - k) + (0,) for k in range(Tr)}
#             seq3 = {(0,) * Tr + (1,)}
#             return seq1.union(seq2.union(seq3))

#         if Tr + 2 <= L <= 2 * Tr + 1:
#             seq1 = {seq + (0,) for seq in recursion(L - 1, Tr)}
#             seq2 = {(0,) * (L - 1) + (1,)}
#             return seq1.union(seq2)

#         if 2 * Tr + 2 <= L <= 3 * Tr + 2:
#             seq1 = {seq + (0,) for seq in recursion(L - 1, Tr)}
#             seq2 = {
#                 (0,) * k + (1,) + (0,) * (L - 2 - Tr) + (1,) + (0,) * (Tr - k) for k in range(Tr)
#             }
#             seq3 = {(0,) * Tr + seq + (0,) * Tr + (1,) for seq in fa_set(L - 2 * Tr - 1, Tr)}
#             return seq1.union(seq2.union(seq3))

#         seq1 = {seq + (0,) for seq in recursion(L - 1, Tr)}

#         seq2 = {
#             (0,) * k + (1,) + (0,) * Tr + seq + (0,) * Tr + (1,) + (0,) * (Tr - k)
#             for k in range(Tr)
#             for seq in fa_set(L - 3 * Tr - 2, Tr)
#         }
#         seq3 = {(0,) * Tr + seq + (0,) * Tr + (1,) for seq in fa_set(L - 2 * Tr - 1, Tr)}
#         return seq1.union(seq2.union(seq3))

#     return recursion(L, Tr)

