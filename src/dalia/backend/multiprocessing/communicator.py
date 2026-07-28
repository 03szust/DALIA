
import numpy as np
from dataclasses import dataclass

from dalia.backend.config import  cupy_version, mpi_version, nccl_version, mpi_cuda_aware
from dalia.backend.datastructures import Matrix
from dalia.backend.multiprocessing.utils import get_active_comm, synchronize_host, synchronize_accelerator

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
        synchronize_host(self.comm)
        synchronize_accelerator()

    def bcast(self, obj: Matrix, root=0):
        """
        Broadcast data from the root process to all other processes within the given communication group.

        Parameters:
        -----------
        data (ArrayLike):
            The data to broadcast.
        root (int), optional:
            The root process. Default is 0.
        """
        if self.comm is not None:
            if  obj.hw_target == "cupy" and not mpi_cuda_aware:
                obj.tohost()
                self.comm.Bcast(obj._data, root=root)
                obj.toaccelerator()
            else:
                self.comm.Bcast(obj._data, root=root)
            return obj
        return obj

    def send(self, obj: Matrix, dest: int, tag: int = 0):
        """
        Send data to a specific process.

        Parameters:
        -----------
        data (ArrayLike):
            The data to send.
        dest (int):
            The rank of the destination process.
        tag (int), optional:
            The message tag. Default is 0.
        """
        if self.comm is not None:
            self.comm.Send(obj._data, dest=dest, tag=tag)

    def recv(self, obj: Matrix, source: int, tag: int = 0):
        """
        Receive data from a specific process.

        Parameters:
        -----------
        source (int):
            The rank of the source process.
        tag (int), optional:
            The message tag. Default is 0.
        """
        if self.comm is not None:
            self.comm.Recv(obj._data, source=source, tag=tag)
        return obj

    def gather(
            self,
            sendbuf: Matrix,
            root: int = 0
        ):
            if self.comm is not None:
                if  sendbuf.hw_target == "cupy" and not mpi_cuda_aware:
                    sendbuf.tohost()
                    recvbuf = np.empty((self.size * sendbuf.shape[0], sendbuf.shape[1]))
                    self.comm.Gather(sendbuf._data, recvbuf, root)
                    recvbuf = recvbuf.reshape(self.size, sendbuf.shape[0], sendbuf.shape[1])
                    sendbuf.toaccelerator()
                    return 
                else:
                    #TODO: accelerator support
                    if self.rank == root:
                        recvbuf = np.empty((self.size * sendbuf.shape[0], sendbuf.shape[1]))
                        self.comm.Gather(sendbuf._data, recvbuf, root)
                        recvbuf = np.split(recvbuf, self.size, axis=0)
                        out = list()
                        for i in recvbuf:
                            m = type(sendbuf)(i, sendbuf._hw_target, sendbuf._override)
                            m._data = i
                            out.append(m)
                        return out
                    else:
                        self.comm.Gather(sendbuf._data, None, root)
                        return None

    def allgather(
            self,
            sendbuf: Matrix,
        ):
            if self.comm is not None:
                if  sendbuf.hw_target == "cupy" and not mpi_cuda_aware:
                    sendbuf.tohost()
                    recvbuf = np.empty((self.size * sendbuf.shape[0], sendbuf.shape[1]))
                    self.comm.Allgather(sendbuf._data, recvbuf)
                    recvbuf = recvbuf.reshape(self.size, sendbuf.shape[0], sendbuf.shape[1])
                    sendbuf.toaccelerator()
                    return 
                else:
                    #TODO: accelerator support
                    recvbuf = np.empty((self.size * sendbuf.shape[0], sendbuf.shape[1]), dtype=sendbuf._data.dtype)
                    self.comm.Allgather(sendbuf._data, recvbuf)
                    recvbuf = np.split(recvbuf, self.size, axis=0)
                    out = list()
                    for i in recvbuf:
                        m = type(sendbuf)(i, sendbuf._hw_target, sendbuf._override)
                        m._data = i
                        out.append(m)
                    return out

    def allgatherv(
        self,
        sendbuf: Matrix,
    ):
        if self.comm is not None:
            if  sendbuf.hw_target == "cupy" and not mpi_cuda_aware:
                sendbuf.tohost()
                recvbuf = np.empty((self.size * sendbuf.shape[0], sendbuf.shape[1]))
                self.comm.Allgather(sendbuf._data, recvbuf)
                recvbuf = recvbuf.reshape(self.size, sendbuf.shape[0], sendbuf.shape[1])
                sendbuf.toaccelerator()
                return 
            else:
                #TODO: accelerator support
                local_rows = sendbuf.shape[0]
                cols = sendbuf.shape[1]

                row_counts = np.empty(self.size, dtype=np.intc)
                self.comm.Allgather(np.array([local_rows], dtype=np.intc), row_counts)

                mpi_counts = row_counts * cols
                mpi_displs = np.insert(np.cumsum(mpi_counts)[:-1], 0, 0)

                total_rows = int(np.sum(row_counts))
                recvbuf = np.empty((total_rows, cols), dtype=sendbuf._data.dtype)

                self.comm.Allgatherv(
                    sendbuf._data, 
                    [recvbuf, (mpi_counts, mpi_displs), MPI._typedict[recvbuf.dtype.char]]
                )

                split_indices = np.cumsum(row_counts)[:-1]
                split_buffers = np.split(recvbuf, split_indices, axis=0)

                out = list()
                for i in split_buffers:
                    m = type(sendbuf)(i, sendbuf._hw_target, sendbuf._override)
                    m._data = i
                    out.append(m)
                return out
            
    def reduce(
            self,
            recvbuf: Matrix,
            op: str = "sum",
            factor: int = 1,
        ):
        """
        Perform a reduction operation across all processes.

        Parameters:
        -----------
        sendbuf (ArrayLike):
            The buffer to send.
        recvbuf (ArrayLike):
            The buffer to receive.
        op (MPI.Op):
            The reduction operation. Default is MPI.SUM.
        root (int):
            The rank of the root process. Default is 0.
        """
        if self.comm is not None:
            if (
                recvbuf._hw_target == "accelerator"
                and not mpi_cuda_aware
            ):

                recvbuf_comm = recvbuf.tohost()
            else:
                recvbuf_comm = recvbuf

            if op == "sum":
                if self.rank == 0:
                    self.comm.Reduce(MPI.IN_PLACE, recvbuf_comm._data, op=MPI.SUM, root=0)
                    recvbuf_comm._data *= factor
                else:
                    self.comm.Reduce(recvbuf_comm._data, None, op=MPI.SUM, root=0)    

            if (
                recvbuf._hw_target == "accelerator"
                and not mpi_cuda_aware
            ):
                # Check if recvbuff is an array or a scalar
                if recvbuf._data.size > 1:
                    recvbuf._data[:] = recvbuf_comm.toaccelerator()
                else:
                    return recvbuf_comm.toaccelerator()

            
            return recvbuf_comm

    def allreduce(
        self,
        recvbuf: Matrix,
        op: str = "sum",
        factor: int = 1
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
                recvbuf_comm = recvbuf.tohost()
            else:
                recvbuf_comm = recvbuf

            if op == "sum":
                self.comm.Allreduce(MPI.IN_PLACE, recvbuf_comm._data, op=MPI.SUM)
                recvbuf_comm._data *= factor

            if (
                recvbuf.hw_target == "accelerator"
                and not mpi_cuda_aware
            ):
                # Check if recvbuff is an array or a scalar
                if recvbuf._data.size > 1:
                    recvbuf_comm.toaccelerator()
                    recvbuf._data[:] = recvbuf_comm._data
                else:
                    return recvbuf_comm.toaccelerator()

            return recvbuf_comm
        

    def reduce_scatter(
            self,
            recvbuf: Matrix,
            op: str = "sum",
            factor: int = 1,
            recvcounts: np.array = None
        ):
        """
        Perform a reduce-scatter operation across all processes within the given communication group.

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
                recvbuf._hw_target == "accelerator"
                and not mpi_cuda_aware
            ):

                recvbuf_comm = recvbuf.tohost()
            else:
                recvbuf_comm = recvbuf

            if recvcounts is None:
                recvcounts = [recvbuf_comm._data.size // self.size] * self.size

            if op == "sum":
                # TODO: slicing to only get relevant array parts.
                self.comm.Reduce_scatter(MPI.IN_PLACE, recvbuf_comm._data, recvcounts, op=MPI.SUM)
                recvbuf_comm._data *= factor
                    

            if (
                recvbuf._hw_target == "accelerator"
                and not mpi_cuda_aware
            ):
                # Check if recvbuff is an array or a scalar
                if recvbuf._data.size > 1:
                    recvbuf._data[:] = recvbuf_comm.toaccelerator()
                else:
                    return recvbuf_comm.toaccelerator()

            
            return recvbuf_comm

    def split(self, color = 0, key = 0):
        """
        Split the communicator into sub-communicators based on color and key.

        Parameters:
        -----------
        color (int):
            The color of the new communicator. Processes with the same color will be in the same new communicator.
        key (int):
            The rank ordering within the new communicator. Processes with the same key will have the same rank in the new communicator.
        """
        if self.comm is not None:
            active_comm = self.comm
            new_comm_group = self.comm.Split(color, key)
            return active_comm, new_comm_group
        else:
            return self.comm, self.comm

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