from dalia.backend.datastructures import Matrix
from dalia.backend.multiprocessing import Communicator
import numpy as np


comm = Communicator()
data = np.eye(2) * (comm.rank + 1)

matrix = Matrix(data, hw_target = "host")
data = np.eye(2) * (comm.rank + 1)
total = Matrix(data, hw_target = "host")

if comm.rank < comm.size-1:
    if comm.rank == 0:
        comm.send(matrix, dest=comm.rank + 1)
        total = comm.recv(total, source=comm.size - 1)
    else:
        total = comm.recv(total, source=comm.rank - 1)
        comm.send(matrix, dest=comm.rank + 1)
elif comm.rank != 0:
    total = comm.recv(total, source=comm.rank - 1)
    comm.send(matrix, dest=0)

if comm.rank == 0:
    expected = np.eye(2) * (comm.size)
else:
    expected = np.eye(2) * (comm.rank)

assert(np.allclose(total.toarray(), expected))
