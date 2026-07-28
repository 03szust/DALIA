from mpi4py import MPI
from dalia.backend.datastructures import Matrix
from dalia.backend.multiprocessing import Communicator
import numpy as np

comm = Communicator()

data = np.eye(2) * (comm.rank + 1)

matrix = Matrix(data, hw_target = "host")
total = comm.allgather(matrix)

expected = np.eye(2) * (comm.size * (comm.size + 1) // 2)

out = np.zeros((2,2))
for i in total:
    out += i.toarray()

assert(np.allclose(out, expected))