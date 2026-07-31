from mpi4py import MPI
from dalia.backend.datastructures import Matrix
from dalia.backend.multiprocessing import Communicator
import numpy as np


comm = Communicator()
data = np.ones((2,2)) * (comm.rank + 1)
print(data)

matrix = Matrix(data, hw_target = "host")
total = comm.reduce_scatter(matrix)

print(comm.rank)

expected = np.ones((2,2)) * (comm.rank + 1)
expected[0,0] = (comm.size * (comm.size + 1) // 2)

print(comm.rank, total, "\n", expected)
assert(np.allclose(total.toarray(), expected))
