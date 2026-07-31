from mpi4py import MPI
from dalia.backend.datastructures import Matrix
from dalia.backend.multiprocessing import Communicator
import numpy as np


comm = Communicator()
data = np.eye(2) * (comm.rank + 1)

matrix = Matrix(data, hw_target = "host")
total = comm.reduce(matrix)

if comm.rank == 0:
    expected = np.eye(2) * (comm.size * (comm.size + 1) // 2)
else:
    expected = np.eye(2) * (comm.rank + 1)

assert(np.allclose(total.toarray(), expected))
