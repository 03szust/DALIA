
from dalia.backend.datastructures import Matrix
from dalia.backend.multiprocessing import Communicator
import numpy as np

comm = Communicator()

data = np.ones((comm.rank + 1, 2)) * (comm.rank + 1)

matrix = Matrix(data, hw_target = "host")
total = comm.allgatherv(matrix)

expected = np.eye(2) * (comm.size * (comm.size + 1) // 2)

assert len(total) == comm.size


for process_idx, gathered_matrix in enumerate(total):
    expected_rows = process_idx + 1
    expected_shape = (expected_rows, 2)
    
    expected_data = np.ones(expected_shape) * (process_idx + 1)
    
    actual_data = gathered_matrix.toarray()
    
    assert actual_data.shape == expected_shape

    assert np.allclose(actual_data, expected_data)