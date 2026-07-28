from mpi4py import MPI
from dalia.backend.datastructures import Matrix
from dalia.backend.multiprocessing import Communicator
import numpy as np

comm = Communicator()

data = np.eye(2) * (comm.rank + 1)

matrix = Matrix(data, hw_target = "host")
print(comm.rank, matrix)
total = comm.allreduce(matrix)

expected = np.eye(2) * (comm.size * (comm.size + 1) // 2)

print(comm.rank,"\n", data,"\n", matrix,"\n", total,"\n", expected)

if total._data.all() != expected.all():
    raise RuntimeError("Incorrect reduction")