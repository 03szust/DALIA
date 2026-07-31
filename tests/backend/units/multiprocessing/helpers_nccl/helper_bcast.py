from dalia.backend.datastructures import Matrix
from dalia.backend.multiprocessing import Communicator
import numpy as np


comm = Communicator()
data = np.eye(2) * (comm.rank + 1)

matrix = Matrix(data, hw_target = "host")
total = comm.bcast(matrix, root=0)

expected = np.eye(2)

assert(np.allclose(total.toarray(), expected))
