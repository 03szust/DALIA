
import numpy as np
from dataclasses import dataclass

from dalia.backend.config import  cupy_version, mpi_version, nccl_version, mpi_cuda_aware
from dalia.backend.datastructures import Matrix
from dalia.backend.multiprocessing.utils import get_host, get_accelerator, get_active_comm

if cupy_version is not None:
    import cupy as cp

if mpi_version is not None:
    from mpi4py import MPI

if nccl_version is not None:
    import nccl
    import cupy.cuda.nccl as cupy_nccl

"""
Base communicator class for multiprocessing.

Should support:
- no-mpi
- mpi4py
- nccl
- gpu-aware mpi
- gpu no-mpi

"""
@dataclass
class Communicator:
    """Base communicator class for multiprocessing.

    Attributes
    ----------
    rank : int
        Rank of the current process.
    size : int
        Total number of processes in the communicator.
    comm : MPI.Comm or None
        The MPI communicator object, if using MPI. None otherwise.
    """
    rank: int = 0
    size: int = 1
    comm: 'MPI.Comm' = None  # Type hint for MPI communicator

    def __post_init__(self):
        """Initialize the communicator based on available libraries."""
        if mpi_version is not None:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.rank = 0
            self.size = 1
            self.comm = None

    def synchronize(self):
        """Synchronize all processes."""
        self.synchronize_host()
        self.synchronize_accelerator()

    def synchronize_host(self):
        """Synchronize all host processes."""
        if self.comm is not None:
            self.comm.Barrier()

    def synchronize_accelerator(self):
            """Synchronize all accelerator processes."""
            if cupy_version is not None:
                cp.cuda.runtime.deviceSynchronize()

    def broadcast(self, data, root=0):
        """Broadcast data from the root process to all other processes."""
        if self.comm is not None:
            return self.comm.bcast(data, root=root)
        return data

    def gather(self, data, root=0):
        """Gather data from all processes to the root process."""
        if self.comm is not None:
            return self.comm.gather(data, root=root)
        return [data]

    def print_msg(self, *args, **kwargs):
        """
        Print a message from a single process.

        Parameters:
        -----------
        *args:
            Variable length argument list.
        **kwargs:
            Arbitrary keyword arguments.
        """
        if self.rank == 0:
            print(*args, **kwargs)

    def allreduce(
        self,
        recvbuf: Matrix,
        op: str = "sum",
        factor: int = 1,
    ):
        """
        Perform a reduction operation across all processes within the given communication group.

        Parameters:
        -----------
        sendbuf (ArrayLike):
            The buffer to send.
        recvbuf (ArrayLike):
            The buffer to receive.
        op ():
            The reduction operation.
        comm (CommunicatorType), optional:
            The communication group. Default is MPI.COMM_WORLD.
        """
        if self.comm is not None:
            if (
                recvbuf.hw_target == "accelerator"
                and not mpi_cuda_aware
            ):
                recvbuf_comm = get_host(recvbuf)
            else:
                recvbuf_comm = recvbuf

            if op == "sum":
                self.comm.Allreduce(MPI.IN_PLACE, recvbuf_comm, op=MPI.SUM)
                recvbuf_comm *= factor

            if (
                recvbuf.hw_target == "accelerator"
                and not mpi_cuda_aware
            ):
                # Check if recvbuff is an array or a scalar
                if recvbuf.data.size > 1:
                    recvbuf.data[:] = get_accelerator(recvbuf_comm)
                else:
                    return get_accelerator(recvbuf_comm)

            if recvbuf.data.size == 1:
                return recvbuf_comm

    def allgather(
        self,
        obj: Matrix,
    ):
        if self.comm is not None:
            if  obj.hw_target == "cupy" and not mpi_cuda_aware:
                obj_comm = get_host(obj)
                return get_accelerator(np.concatenate(self.comm.allgather(obj_comm)))
            else:
                return self.comm.allgather(obj)

    # TODO: move this to multiprocessing utils

    def smartsplit(
        self,
        n_parallelizable_evaluations: int,
        tag: str,
        min_group_size: int = 1,
    ) -> tuple:
        if self.comm is not None:
            if self.comm.size < min_group_size:
                raise ValueError(
                    f"Initial CommunicatorType size must be at least {min_group_size} to fulfill the split requirements."
                )

            # Checks for compatibility of given comm sizes
            min_comm = get_active_comm(self.comm, min_group_size, tag="minimum_comm")
            active_comm = get_active_comm(
                min_comm, n_parallelizable_evaluations * min_group_size, tag
            )
            rank = active_comm.Get_rank()
            size = active_comm.size

            # Compute the group size, given its minimum
            group_size = size // n_parallelizable_evaluations
            if group_size < min_group_size:
                group_size = min_group_size

            # Split the CommunicatorType
            color_new_group = rank // group_size
            key_new_group = rank
            comm_new_group = active_comm.comm.Split(color_new_group, key_new_group)
        else:
            active_comm = self.comm
            comm_new_group = self.comm
            color_new_group = 0

        return active_comm, comm_new_group, color_new_group