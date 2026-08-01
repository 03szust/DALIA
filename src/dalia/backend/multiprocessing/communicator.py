
import numpy as np
from dataclasses import dataclass, field

from dalia.backend.config import  cupy_version, mpi_version, nccl_version, mpi_cuda_aware, compare_version
from dalia.backend.datastructures import Matrix
from dalia.backend.multiprocessing.utils import get_active_comm, synchronize_host, synchronize_accelerator, get_mpi_operation, get_accelerator_count, comm_decider, get_nccl_operation, get_nccl_dtype

if cupy_version is not None:
    import cupy as cp

    if nccl_version is not None:
        import nccl
        import cupy.cuda.nccl as cupy_nccl

if mpi_version is not None:
    from mpi4py import MPI



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

    Variables:
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

    nccl_comms: dict = field(default_factory=dict) # list of nccl communicators
    nccl_id: bytes = None

    no_mpi: bool = False
    no_nccl: bool = False

    _skip_init: bool = field(default=False, repr=False)

    def __post_init__(self):
        """Initialize the communicator based on available libraries."""
        if self._skip_init:
            return
        
        if mpi_version is not None or self.no_mpi:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.rank = 0
            self.size = 1
            self.comm = None
            self.no_mpi = True

        if nccl_version is not None or self.no_nccl:
            
            accelerator_count, self.nccl_accelerators = get_accelerator_count()
            if self.comm is None:
                self.nccl_id = cupy_nccl.get_unique_id()
                self.nccl_comms = cupy_nccl.initAll(self.nccl_accelerators)
            else:
                if self.rank == 0:
                    self.nccl_id = cupy_nccl.get_unique_id()
                self.nccl_id = self.comm.bcast(self.nccl_id, root=0)

                local_accelerators = []
                for accelerator_id in self.nccl_accelerators:
                    if accelerator_id % self.size == self.rank:
                        local_accelerators.append(accelerator_id)

                for accelerator_id in local_accelerators:
                    cp.cuda.Device(accelerator_id).use()
                    nccl_comm = cupy_nccl.NcclCommunicator(accelerator_count, self.nccl_id, accelerator_id)
                    self.nccl_comms[accelerator_id] = nccl_comm

                self.comm.Barrier()
        else:
            self.no_nccl = True

    def synchronize(self):
        """Synchronize all processes."""
        synchronize_host(self.comm)
        synchronize_accelerator()

    def bcast(
            self, 
            obj: Matrix, 
            root = 0,
            comm_type = "auto",
            stream = None,
            accelerator_id = None
    ):
        """
        Broadcast data from the root process to all other processes within the given communication group.

        Args:
            obj (Matrix):
                The data to broadcast.
            root (int), optional:
                The root process. Default is 0.
        Returns:
            obj (Matrix): 
                The data to broadcast.
        """
        is_mpi, is_nccl = comm_decider(comm_type, obj)

        if self.comm is not None and is_mpi:
            if  obj._hw_target == "accelerator" and not mpi_cuda_aware:
                obj.tohost()
                self.comm.Bcast(obj._data, root=root)
                obj.toaccelerator()
            else:
                self.comm.Bcast(obj._data, root=root)
            return obj
        elif len(self.nccl_comms) and is_nccl:
            if accelerator_id is None:
                if len(self.nccl_comms) == 1:
                    nccl_comm = self.nccl_comms[0]
                else:
                    raise ValueError(f"No accelerator specified even thoug multiple are available")
            else:
                nccl_comm = self.nccl_comms[accelerator_id]
            if stream is None:
                stream = cp.cuda.Stream.null
            if  obj._hw_target == "host" and not mpi_cuda_aware:
                obj.toaccelerator()
                nccl_comm.bcast(obj._data.data.ptr, obj._data.size, obj._data.dtype, root, stream.ptr)
                obj.tohost()
            else:
                nccl_comm.bcast(obj._data.data.ptr, obj._data.size, obj._data.dtype, root, stream.ptr)
        return obj

    def send(
            self,
            obj: Matrix, 
            dest: int, tag: 
            int = 0, 
            comm_type = "auto",
            stream = None,
            accelerator_id = None
    ):
        """
        Send data to a specific process.

        Args:
            obj (Matrix):
                The data to send.
            dest (int):
                The rank of the destination process.
            tag (int), optional:
                The message tag. Default is 0.
        Returns:
            obj (Matrix):
                The data to send.
        """
        is_mpi, is_nccl = comm_decider(comm_type, obj)

        if self.comm is not None and is_mpi:
            if  obj._hw_target == "accelerator" and not mpi_cuda_aware:
                obj.tohost()
                self.comm.Send(obj._data, dest=dest, tag=tag)
                obj.toaccelerator()
            else:
                self.comm.Send(obj._data, dest=dest, tag=tag)
        elif len(self.nccl_comms) and is_nccl:
            if accelerator_id is None:
                if len(self.nccl_comms) == 1:
                    nccl_comm = self.nccl_comms[0]
                else:
                    raise ValueError(f"No accelerator specified even thoug multiple are available")
            else:
                nccl_comm = self.nccl_comms[accelerator_id]
            if stream is None:
                stream = cp.cuda.Stream.null
            if  obj._hw_target == "host" and not mpi_cuda_aware:
                obj.toaccelerator()
                nccl_comm.send(obj._data.data.ptr, obj._data.size, obj._data.dtype, dest, stream.ptr)
                obj.tohost()
            else:
                nccl_comm.send(obj._data.data.ptr, obj._data.size, obj._data.dtype, dest, stream.ptr)
        return obj

    def recv(
            self,
            obj: Matrix, 
            source: int, 
            tag: int = 0, 
            comm_type = "auto",
            stream = None
    ):
        """
        Receive data from a specific process.

        Args:
            obj (Matrix):
                The buffer to recieve the data.
            source (int):
                The rank of the source process.
            tag (int), optional:
                The message tag. Default is 0.
        Returns:
            obj (Matrix):
                The buffer to recieve the data.
        """
        is_mpi, is_nccl = comm_decider(comm_type, obj)

        if self.comm is not None and is_mpi:
            if  obj._hw_target == "accelerator" and not mpi_cuda_aware:
                obj.tohost()
                self.comm.Recv(obj._data, source=source, tag=tag)
                obj.toaccelerator()
            else:
                self.comm.Recv(obj._data, source=source, tag=tag)
        elif len(self.nccl_comms) and is_nccl:
            if stream is None:
                stream = cp.cuda.Stream.null
            if  obj._hw_target == "host" and not mpi_cuda_aware:
                obj.toaccelerator()
                nccl_dtype = get_nccl_dtype(obj._data.dtype)
                self.comm.recv(obj._data.data.ptr, obj._data.size, nccl_dtype, source, stream.ptr)
                obj.tohost()
            else:
                self.comm.recv(obj._data.data.ptr, obj._data.size, obj._data.dtype, source, stream.ptr)
        return obj

    def gather(
            self,
            sendbuf: Matrix,
            root: int = 0, 
            comm_type = "auto"
    ):
        """
        Gather data to a specific process
        
        Args:
            sendbuf (Matrix):
                The data to send.
            root (int):
                The rank of the root process.
        Returns:
            out (list) on None:
                The recieved data on root process.
                None on not root processes or if there is no communicator.
        """
        is_mpi, is_nccl = comm_decider(comm_type, sendbuf)

        if self.comm is not None and is_mpi:
            xp = np
            hw_target = sendbuf._hw_target
            if  sendbuf._hw_target == "accelerator":
                if  mpi_cuda_aware:    
                    xp = cp
                else:
                    sendbuf.tohost()

            if self.rank == root:
                recvbuf = xp.empty((self.size * sendbuf.shape[0], sendbuf.shape[1]))
                self.comm.Gather(sendbuf._data, recvbuf, root)
                recvbuf = xp.split(recvbuf, self.size, axis=0)
                out = list()
                for i in recvbuf:
                    m = type(sendbuf)(i, hw_target, sendbuf._override)
                    m._data = i
                    out.append(m)
                return out
            else:
                self.comm.Gather(sendbuf._data, None, root)
                return None
        elif len(self.nccl_comms) and is_nccl:
            # TODO: maybe swap to normal nccl
            raise NotImplementedError(f"cupy nccl does not implement Gather")
        return None

    def allgather(
            self,
            sendbuf: Matrix,
            comm_type = "auto",
            stream = None,
            accelerator_id = None
    ):
        """
        Gather data to all processes
        
        Args:
            sendbuf (Matrix):
                The data to send.
        Returns:
            out (list) or None:
                The recieved data.
                None if there is no communicator.
        """
        is_mpi, is_nccl = comm_decider(comm_type, sendbuf)

        if self.comm is not None and is_mpi:
            xp = np
            hw_target = sendbuf._hw_target
            if  sendbuf._hw_target == "accelerator":
                if  mpi_cuda_aware:    
                    xp = cp
                else:
                    sendbuf.tohost()

            recvbuf = xp.empty((self.size * sendbuf.shape[0], sendbuf.shape[1]), dtype=sendbuf._data.dtype)
            self.comm.Allgather(sendbuf._data, recvbuf)
            recvbuf = xp.split(recvbuf, self.size, axis=0)
            out = list()
            for i in recvbuf:
                m = type(sendbuf)(i, hw_target, sendbuf._override)
                m._data = i
                out.append(m)
            return out
        elif len(self.nccl_comms) and is_nccl:
            if accelerator_id is None:
                if len(self.nccl_comms) == 1:
                    nccl_comm = self.nccl_comms[0]
                else:
                    raise ValueError(f"No accelerator specified even thoug multiple are available")
            else:
                if accelerator_id in self.nccl_comms:
                    nccl_comm = self.nccl_comms[accelerator_id]
                else:
                    raise ValueError(f"Accelerator {accelerator_id} is not available in the communicator")
            if stream is None:
                stream = cp.cuda.Stream.null
            hw_target = sendbuf._hw_target
            if  sendbuf._hw_target == "host":
                sendbuf.toaccelerator()
            nccl_dtype = get_nccl_dtype(sendbuf._data.dtype)
            recvbuf = cp.empty((self.size * sendbuf.shape[0], sendbuf.shape[1]), dtype=sendbuf._data.dtype)
            nccl_comm.allGather(sendbuf._data.data.ptr, recvbuf.data.ptr, sendbuf._data.size, nccl_dtype, stream.ptr)
            recvbuf = cp.split(recvbuf, self.size, axis=0)
            out = list()
            for i in recvbuf:
                m = type(sendbuf)(i, hw_target, sendbuf._override)
                m._data = i
                out.append(m)
            return out
        return None

    def allgatherv(
            self,
            sendbuf: Matrix,
            comm_type = "auto"
    ):
        """
        Gather data of different sizes to all processes
        
        Args:
            sendbuf (Matrix):
                The data to send.
        Returns:
            out (list) ot None:
                The recieved data
                None if there is no communicator.
        """
        is_mpi, is_nccl = comm_decider(comm_type, sendbuf)

        if self.comm is not None and is_mpi:
            xp = np
            hw_target = sendbuf._hw_target
            if  sendbuf._hw_target == "accelerator":
                if  mpi_cuda_aware:    
                    xp = cp
                else:
                    sendbuf.tohost()

            local_rows = sendbuf.shape[0]
            cols = sendbuf.shape[1]

            row_counts = np.empty(self.size, dtype=np.intc)
            self.comm.Allgather(np.array([local_rows], dtype=np.intc), row_counts)

            mpi_counts = row_counts * cols
            mpi_displs = np.insert(np.cumsum(mpi_counts)[:-1], 0, 0)

            total_rows = int(np.sum(row_counts))
            recvbuf = xp.empty((total_rows, cols), dtype=sendbuf._data.dtype)

            self.comm.Allgatherv(
                sendbuf._data, 
                [recvbuf, (mpi_counts, mpi_displs), MPI._typedict[recvbuf.dtype.char]]
            )

            split_indices = np.cumsum(row_counts)[:-1]
            split_buffers = np.split(recvbuf, split_indices, axis=0)

            out = list()
            for i in split_buffers:
                m = type(sendbuf)(i, hw_target, sendbuf._override)
                m._data = i
                out.append(m)
            return out
        elif len(self.nccl_comms) and is_nccl:
            raise NotImplementedError(f"nccl does not implement Allgatherv")
        return None
            
    def reduce(
            self,
            recvbuf: Matrix,
            op: str = "sum",
            root: int = 0,
            factor: int = 1,
            comm_type = "auto",
            stream = None,
            accelerator_id = None
        ):
        """
        Perform a reduction operation to a specific processes.

        Args:
            recvbuf (Matrix):
                The buffer to send and receive data.
            op (string)):
                The reduction operation. Default is sum.
            root (int):
                The rank of the root process. Default is 0.
            factor (int):
                The factor to multiply the recieved array with. Default is 1.
        Returns:
            recvbuf (Matrix):
                The buffer to send and receive data.
        """
        is_mpi, is_nccl = comm_decider(comm_type, recvbuf)

        if self.comm is not None and is_mpi:
            if recvbuf._hw_target == "accelerator" and not mpi_cuda_aware:
                recvbuf.tohost()

            op = get_mpi_operation(op)
            
            if self.rank == root:
                self.comm.Reduce(MPI.IN_PLACE, recvbuf._data, op=op, root=root)
                recvbuf._data *= factor
            else:
                self.comm.Reduce(recvbuf._data, None, op=op, root=root)    

            if recvbuf._hw_target == "accelerator"and not mpi_cuda_aware:
                recvbuf.toaccelerator()
            
        elif len(self.nccl_comms) and is_nccl:
            if accelerator_id is None:
                if len(self.nccl_comms) == 1:
                    nccl_comm = self.nccl_comms[0]
                else:
                    raise ValueError(f"No accelerator specified even thoug multiple are available")
            else:
                nccl_comm = self.nccl_comms[accelerator_id]
            if stream is None:
                stream = cp.cuda.Stream.null

            if recvbuf._hw_target == "host" and not mpi_cuda_aware:
                recvbuf.toaccelerator()

            nccl_dtype = get_nccl_dtype(recvbuf._data.dtype)
            op = get_nccl_operation(op)
            
            if self.rank == root:
                self.comm.reduce(recvbuf._data.data.ptr, recvbuf._data.data.ptr, recvbuf._data.size, nccl_dtype, op, root, stream.ptr)
                recvbuf._data *= factor
            else:
                self.comm.reduce(recvbuf._data.data.ptr, recvbuf._data.data.ptr, recvbuf._data.size, nccl_dtype, op, root, stream.ptr)

            if recvbuf._hw_target == "host"and not mpi_cuda_aware:
                recvbuf.tohost()
        return recvbuf    
        

    def allreduce(
            self,
            recvbuf: Matrix,
            op: str = "sum",
            factor: int = 1,
            comm_type = "auto",
            stream = None,
            accelerator_id = None
    ):
        """
        Perform a reduction operation across all processes.

        Args:
            recvbuf (Matrix):
                The buffer to send and receive data.
            op (string)):
                The reduction operation. Default is sum.
            factor (int):
                The factor to multiply the recieved array with. Default is 1.
        Returns:
            recvbuf (Matrix):
                The buffer to send and receive data.
        """
        is_mpi, is_nccl = comm_decider(comm_type, recvbuf)

        if self.comm is not None and is_mpi:
            
            if recvbuf._hw_target == "accelerator" and not mpi_cuda_aware:
                recvbuf.tohost()

            op = get_mpi_operation(op)

            self.comm.Allreduce(MPI.IN_PLACE, recvbuf._data, op=op)
            recvbuf._data *= factor

            if recvbuf._hw_target == "accelerator" and not mpi_cuda_aware:
                recvbuf.toaccelerator()
        elif len(self.nccl_comms) and is_nccl:
            if accelerator_id is None:
                if len(self.nccl_comms) == 1:
                    nccl_comm = self.nccl_comms[0]
                else:
                    raise ValueError(f"No accelerator specified even thoug multiple are available")
            else:
                nccl_comm = self.nccl_comms[accelerator_id]
            if stream is None:
                stream = cp.cuda.Stream.null

            if recvbuf._hw_target == "host" and not mpi_cuda_aware:
                recvbuf.toaccelerator()

            nccl_dtype = get_nccl_dtype(recvbuf._data.dtype)
            op = get_nccl_operation(op)

            nccl_comm.allReduce(recvbuf._data.data.ptr, recvbuf._data.data.ptr, recvbuf._data.size, nccl_dtype, op, stream.ptr)
            recvbuf._data *= factor

            if recvbuf._hw_target == "host"and not mpi_cuda_aware:
                recvbuf.tohost()
        return recvbuf
        

    def reduce_scatter(
            self,
            recvbuf: Matrix,
            op: str = "sum",
            factor: int = 1,
            recvcounts: np.array = None,
            comm_type = "auto",
            stream = None,
            accelerator_id = None
        ):
        """
        Perform a reduce-scatter operation across all processes within the given communication group.

        Args:
            recvbuf (Matrix):
                The buffer to send and receive data.
            op (string)):
                The reduction operation. Default is sum.
            factor (int):
                The factor to multiply the recieved array with. Default is 1.
            recvcounts (np.array):
                scatter pattern
        Returns:
            recvbuf (Matrix):
                The buffer to send and receive data.
        Raises:
            ValueError:
                The scatter pattern does not match the communicator or the data.
        """
        is_mpi, is_nccl = comm_decider(comm_type, recvbuf)

        if self.comm is not None and is_mpi:
            
            if recvbuf._hw_target == "accelerator" and not mpi_cuda_aware:
                recvbuf.tohost()

            op = get_mpi_operation(op)

            if recvcounts is None:  
                recvcounts = np.full(self.size, recvbuf._data.size // self.size, dtype=np.intc)
                recvcounts[:recvbuf._data.size % self.size] += 1
            elif recvcounts.size != self.size:
                raise ValueError (f"The size of recvcounts ({recvcounts.size}) does not match the comm size ({self.size})")
            elif np.sum(recvcounts) != recvbuf._data.size:
                raise ValueError (f"The sum of recvcounts ({np.sum(recvcounts)}) does not match the data size ({recvbuf._data.size})")

            # TODO: slicing to only get relevant array parts.
            self.comm.Reduce_scatter(MPI.IN_PLACE, recvbuf._data, recvcounts, op=MPI.SUM)
            recvbuf._data *= factor
            
            if recvbuf._hw_target == "accelerator" and not mpi_cuda_aware:
                recvbuf.toaccelerator()
        elif len(self.nccl_comms) and is_nccl:
            if accelerator_id is None:
                if len(self.nccl_comms) == 1:
                    nccl_comm = self.nccl_comms[0]
                else:
                    raise ValueError(f"No accelerator specified even thoug multiple are available")
            else:
                nccl_comm = self.nccl_comms[accelerator_id]
            if stream is None:
                stream = cp.cuda.Stream.null

            if recvbuf._hw_target == "host":
                recvbuf.toaccelerator()

            nccl_dtype = get_nccl_dtype(recvbuf._data.dtype)
            op = get_nccl_operation(op)

            if recvcounts is None:  
                recvcounts = cp.full(self.size, recvbuf._data.size // self.size, dtype=cp.intc)
                recvcounts[:recvbuf._data.size % self.size] += 1
            elif recvcounts.size != self.size:
                raise ValueError (f"The size of recvcounts ({recvcounts.size}) does not match the comm size ({self.size})")
            elif cp.sum(recvcounts) != recvbuf._data.size:
                raise ValueError (f"The sum of recvcounts ({np.sum(recvcounts)}) does not match the data size ({recvbuf._data.size})")

            # TODO: slicing to only get relevant array parts.
            nccl_comm.reduceScatter(recvbuf._data.data.ptr, recvbuf._data.data.ptr, recvbuf._data.size, nccl_dtype, op, stream.ptr)
            recvbuf._data *= factor
            
            if recvbuf._hw_target == "host":
                recvbuf.tohost()
        return recvbuf

    def split(
            self, 
            color = 0, 
            key = 0,
            
    ):
        """
        Split the communicator into sub-communicators based on color and key.

        Args:
            color (int):
                The color of the new communicator. Processes with the same color will be in the same new communicator.
            key (int):
                The rank ordering within the new communicator. Processes with the same key will have the same rank in the new communicator.
        """
        new_comm = Communicator(_skip_init=True)

        if self.comm is not None:
            new_comm_group = self.comm.Split(color, key)

            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()

            new_comm.comm = new_comm_group
            new_comm.rank = new_comm.comm.Get_rank()
            new_comm.size = new_comm.comm.Get_size()
            new_comm.nccl_comms = []

        if nccl_version is not None and not self.no_nccl:

            if compare_version(nccl_version, "2.18.1"):
                for parent_nccl_comm in self.nccl_comms:
                    gpu_id = parent_nccl_comm.device_id()
                    cp.cuda.Device(gpu_id).use()
                    
                    sub_nccl_comm = parent_nccl_comm.commSplit(color, key)
                    
                    if sub_nccl_comm is not None:
                        new_comm.nccl_comms.append(sub_nccl_comm)

                new_comm.comm.Barrier()

        return self, new_comm
        

    def smartsplit(
        self,
        n_parallelizable_evaluations: int,
        tag: str,
        min_group_size: int = 1,
    ) -> tuple:
        # TODO: smartsplit has been directly adapted from DALIA V1. Documentation is not clear on what smartsplit does.
        # TODO: since smartsplit is out of scope of the project, there is no nccl implementation for smartsplit.
        """
                Executes an adapted smartsplit from DALIA V1.
        
                Args:
                    n_parallelizable_evaluations (int):
                        The number of parallelizable evaluations.
                    tag (str):
                       Tag of the communicator.
                    min_group_size (int):
                        The minimal group size.
                Returns:
                    tuple (Communicator, Communicator, int):
                        The active communicator,
                        The new communicator,
                        The color of the new communicator
                """
        if self.comm is not None:
            if self.comm.size < min_group_size:
                raise ValueError(
                    f"Initial CommunicatorType size must be at least {min_group_size} to fulfill the split requirements."
                )

            # Checks for compatibility of given comm sizes
            min_comm = get_active_comm(self.comm, min_group_size, tag="minimum_comm")
            active_comm_obj = get_active_comm(
                min_comm, n_parallelizable_evaluations * min_group_size, tag
            )
            rank = active_comm_obj.Get_rank()
            size = active_comm_obj.size

            # Compute the group size, given its minimum
            group_size = size // n_parallelizable_evaluations
            if group_size < min_group_size:
                group_size = min_group_size

            # Split the CommunicatorType
            color_new_group = rank // group_size
            key_new_group = rank
            comm_new_group_obj = active_comm_obj.comm.Split(color_new_group, key_new_group)

            active_comm = Communicator()
            active_comm.comm = active_comm_obj
            active_comm.rank = active_comm.comm.Get_rank()
            active_comm.size = active_comm.comm.Get_size()

            comm_new_group = Communicator()
            comm_new_group.comm = comm_new_group_obj
            comm_new_group.rank = comm_new_group.comm.Get_rank()
            comm_new_group.size = comm_new_group.comm.Get_size()

        else:
            active_comm = self
            comm_new_group = self
            color_new_group = 0

        return active_comm, comm_new_group, color_new_group

    def print_msg(self, *args, root=0, **kwargs):
        """
        Print a message from a single process.

        Args:
            *args:
                Variable length argument list.
            root:
                process to print from. Default is 0
            **kwargs:
                Arbitrary keyword arguments.
        """
        if self.rank == root:
            print(*args, **kwargs)