import contextlib
import torch
from collections import defaultdict
from torch.autograd.graph import saved_tensors_hooks
from enum import Enum
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
import threading
import gc
from megatron.core import parallel_state
from contextlib import nullcontext

def checksum(tensor):
    with torch.no_grad():
        if tensor.dtype == torch.half:
            return torch.mean(tensor * tensor).sum().item()
        else:
            return 0


def is_a_view(x, y):
    return x.storage().data_ptr() == y.storage().data_ptr() and x.storage_offset() == y.storage_offset() and x.numel() == y.numel()


def tensor_info(tensor):
    return (tensor.shape, tensor.layout, tensor.dtype, tensor.stride())


def save_rng_states():
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    return torch.get_rng_state(), torch.cuda.get_rng_state(), get_cuda_rng_tracker().get_states()


def restore_rng_states(states):
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, _set_cuda_rng_state
    torch.set_rng_state(states[0])
    _set_cuda_rng_state(states[1])
    get_cuda_rng_tracker().set_states(states[2])


class PartialRecompute(saved_tensors_hooks):
    class RecomputeSaveType(Enum):
        PASS_THROUGH = 1
        RECOMPUTE = 2

    def _save_tensor(self, tensor):
        if self._next_recompute_tensor is not None and is_a_view(tensor, self._next_recompute_tensor[0]):
            packed = self._next_recompute_tensor[1:]
            self._next_recompute_tensor = None
            return PartialRecompute.RecomputeSaveType.RECOMPUTE, packed
        return PartialRecompute.RecomputeSaveType.PASS_THROUGH, tensor

    def _resume_tensor(self, packed):
        type, info = packed
        if type == PartialRecompute.RecomputeSaveType.RECOMPUTE:
            parents, function, view_size, rng_states = info
            with torch.no_grad():
                if rng_states is not None:
                    current_rng_states = save_rng_states()
                    restore_rng_states(rng_states)
                # context = self.context if self.context is not None else nullcontext
                # with context:
                r = function(*parents)
                if view_size is not None:
                    r = r.view(*view_size)
                self.context = None
                self.kwargs = None
                if rng_states is not None:
                    restore_rng_states(current_rng_states)
            if self.bias != None:
                r = (r, self.bias)
                self.bias =None
            return r
        return info

    def __init__(self):
        self._next_recompute_tensor = None
        self.bias = None
        self.kwargs = None
        self.context = None
        super().__init__(self._save_tensor, self._resume_tensor)

    def _recompute_tensor(self, tensor, parents, function, view_size=None, rng_states=None):
        assert self._next_recompute_tensor is None
        if isinstance(tensor, tuple):
            assert self.bias == None
            self.bias = tensor[1]
            tensor = tensor[0]
        
        # if isinstance(parents[-2], dict):
        #     assert self.kwargs == None
        #     self.kwargs = parents[-2]
        # if parents[-1] != None:
        #     assert self.context == None
        #     self.context = parents[-1]
        self._next_recompute_tensor = (tensor, parents, function, view_size, rng_states)


partial_recompute = PartialRecompute()

class ActivationStore(saved_tensors_hooks):
    @classmethod
    def recompute_tensor(cls, tensor, parents, function, view_size=None, rng_states=None):
        return partial_recompute._recompute_tensor(tensor, parents, function, view_size, rng_states)

    def __enter__(self):
        assert not hasattr(
            ActivationStore, '_current_activation_store') or ActivationStore._current_activation_store is None, "Nested offload not supported"
        ActivationStore._current_activation_store = self
        return super().__enter__()

    def __exit__(self, *args):
        super().__exit__(*args)
        ActivationStore._current_activation_store = None

    class State(Enum):
        NEW = 0
        SAVING = 1
        OFFLOADED = 2
        OFFLOAD_RELEASED = 3
        RESUME_PREPARED = 4
        RESUMED = 5
        RESUME_USED = 6
        RESUME_RELEASED = 7

    def _change_state(self, from_state, to_state):
        with self._state_lock:
            if isinstance(from_state, set):
                assert self._state in from_state
            else:
                assert self._state == from_state, f"from_state {from_state} is not equal to current state {self._state}"
            self._state = to_state

    class SaveType(Enum):
        OFFLOAD = 1
        PASS_THROUGH = 2
        RECOMPUTE = 3
        ALIAS = 4

    def _save_tensor(self, tensor):
        assert not self._offloaded
        # print(f"this tensor contiguous is {tensor.is_contiguous()}, the shape is {tensor.shape}")
        # print(f"Received tensor type: {type(tensor)}, is Parameter: {isinstance(tensor, torch.nn.parameter.Parameter)}")
        if not tensor.is_contiguous():
            # print(f"type is {type(tensor)}")
            tensor = tensor.contiguous()
        self._change_state({ActivationStore.State.NEW, ActivationStore.State.SAVING},
                           ActivationStore.State.SAVING)
        if isinstance(tensor, torch.nn.parameter.Parameter):
            return ActivationStore.SaveType.PASS_THROUGH, tensor
        if tensor.numel() <= 1024:
            return ActivationStore.SaveType.PASS_THROUGH, tensor
        if not tensor.is_contiguous():
            return ActivationStore.SaveType.PASS_THROUGH, tensor
        recompute = partial_recompute._save_tensor(tensor)
        if recompute[0] == PartialRecompute.RecomputeSaveType.RECOMPUTE:
            (parents, function, view_size, rng_states) = recompute[1]
            parent_handles = [self._save_tensor(x) for x in parents]
            # parent_handles = [x for x in parents]
            return ActivationStore.SaveType.RECOMPUTE, (parent_handles, function, view_size, rng_states)
        if self.is_a_view_opti:
            # 优化：使用哈希表查找，从 O(n) 降到 O(1)
            storage_ptr = tensor.storage().data_ptr()
            tensor_offset = tensor.storage_offset()
            tensor_numel = tensor.numel()

            if storage_ptr in self._storage_index_map:
                # 查找所有具有相同 storage 的候选项
                for index, stored_offset, stored_numel in self._storage_index_map[storage_ptr]:
                    if tensor_offset == stored_offset and tensor_numel == stored_numel:
                        # 找到匹配的 view
                        stored_tensor = self._gpu_store[index]
                        offset = tensor.storage_offset() - stored_tensor.storage_offset()
                        stride = tensor.stride()
                        shape = tensor.shape
                        return ActivationStore.SaveType.ALIAS, (tensor.dtype, index, shape, stride, offset)

            # 不是 view，需要保存新的 tensor
            new_index = len(self._gpu_store)
            self._gpu_store.append(tensor.data)

            # 更新索引字典
            if storage_ptr not in self._storage_index_map:
                self._storage_index_map[storage_ptr] = []
            self._storage_index_map[storage_ptr].append((new_index, tensor_offset, tensor_numel))
        else:
            for index, stored_tensor in enumerate(self._gpu_store):
                if is_a_view(tensor, stored_tensor):
                    offset = tensor.storage_offset() - stored_tensor.storage_offset()
                    stride = tensor.stride()
                    shape = tensor.shape
                    return ActivationStore.SaveType.ALIAS, (tensor.dtype, index, shape, stride, offset)
            self._gpu_store.append(tensor.data)
        if (len(self._offload_tensor_info) < len(self._gpu_store)):
            self._offload_tensor_info.append(tensor_info(tensor))
        else:
            assert (self._offload_tensor_info[len(self._gpu_store) - 1] == tensor_info(tensor))
        self._save_event.record()
        # print(f"rank {torch.distributed.get_rank()} Saving tensor id {len(self._gpu_store) - 1} {id(tensor)} {tensor.shape}, dtype {tensor.dtype}, device {tensor.device} storage {tensor.storage().data_ptr()}")
        # if len(self._gpu_store)>=44:
        #     if torch.distributed.get_rank()==0:
        #         print("------------------------------------------")
        #         print(len(self._gpu_store))
        #         for x in self._gpu_store:
        #             print(f"{x.shape} {x.dtype}")
        #         print("------------------------------------------")
        # if self._gpu_store[-1].shape==torch.Size([4096, 1, 4096]):
        #     print("123")
        return (ActivationStore.SaveType.OFFLOAD, len(self._gpu_store) - 1)

    def _resume_tensor(self, packed, remove_used=True):
        assert not self._offloaded
        type, info = packed
        self._change_state({ActivationStore.State.RESUMED,
                           ActivationStore.State.RESUME_USED}, ActivationStore.State.RESUME_USED)
        if type == ActivationStore.SaveType.PASS_THROUGH:
            # print(f"In Resume rank:{torch.distributed.get_rank()} main_grad:{hasattr(info, 'main_grad')} Received tensor type: {info.__class__}, is Parameter: {isinstance(info, torch.nn.parameter.Parameter)}")
            return info
        if type == ActivationStore.SaveType.RECOMPUTE:
            p_infos, function, view_size, rng_states = info
            parents = [self._resume_tensor(x, remove_used=False) for x in p_infos]
            # parents = [x for x in p_infos]
            return partial_recompute._resume_tensor((PartialRecompute.RecomputeSaveType.RECOMPUTE, (parents, function, view_size, rng_states)))
        if packed[0] == ActivationStore.SaveType.ALIAS:
            dtype, index, shape, stride, offset = packed[1]

            # 第一次调用时统计wait时间
            if self._first_resume_tensor_call:
                self._resume_wait_start_event.record()

            self._resume_event.wait()

            if self._first_resume_tensor_call:
                self._resume_wait_end_event.record()
                self._first_resume_tensor_call = False

                # 同步并计算等待时间和H2D带宽
                self._resume_wait_end_event.synchronize()
                wait_time_ms = self._resume_wait_start_event.elapsed_time(
                    self._resume_wait_end_event)
                h2d_time_ms = self._resume_start_event.elapsed_time(self._resume_end_event)
                h2d_time_s = h2d_time_ms / 1000.0
                bandwidth_gbps = (self._total_bytes_resumed / 1e9) / \
                    h2d_time_s if h2d_time_s > 0 else 0

                print(f"rank {torch.distributed.get_rank()} H2D带宽: {bandwidth_gbps:.2f} GB/s "
                      f"({self._total_bytes_resumed / 1e9:.3f} GB / {h2d_time_ms:.2f} ms), "
                      f"alias 默认流等待H2D时间: {wait_time_ms:.2f} ms")

            # print(f"rank {torch.distributed.get_rank()} Resuming alias tensor id {index} {shape}, offset {offset}")
            bin, o = self.index_offset[index]
            return torch.as_strided(self._continuous_gpu_buffer[dtype][bin], shape, stride, o + offset)
        assert type == ActivationStore.SaveType.OFFLOAD

        # 第一次调用时统计wait时间
        if self._first_resume_tensor_call:
            self._resume_wait_start_event.record()

        self._resume_event.wait()

        if self._first_resume_tensor_call:
            self._resume_wait_end_event.record()
            self._first_resume_tensor_call = False

            # 同步并计算等待时间和H2D带宽
            self._resume_wait_end_event.synchronize()
            wait_time_ms = self._resume_wait_start_event.elapsed_time(self._resume_wait_end_event)
            h2d_time_ms = self._resume_start_event.elapsed_time(self._resume_end_event)
            h2d_time_s = h2d_time_ms / 1000.0
            bandwidth_gbps = (self._total_bytes_resumed / 1e9) / h2d_time_s if h2d_time_s > 0 else 0

            print(f"rank {torch.distributed.get_rank()} H2D带宽: {bandwidth_gbps:.2f} GB/s "
                  f"({self._total_bytes_resumed / 1e9:.3f} GB / {h2d_time_ms:.2f} ms), "
                  f"offload 默认流等待H2D时间: {wait_time_ms:.2f} ms")

        index = info
        ret = self._gpu_store[index]
        self._gpu_store[index] = None
        if remove_used:
            shape, layout, dtype, stride = self._offload_tensor_info[index]
            bin, offset = self.index_offset[index]
            all_freed = True
            for (i, (b, o)) in enumerate(self.index_offset):
                if self._gpu_store[i] is not None and b == bin and self._offload_tensor_info[i][2] == dtype:
                    all_freed = False
                    break
            if all_freed:
                self._continuous_gpu_buffer[dtype][bin] = None
        # print(f"rank {torch.distributed.get_rank()} Resuming tensor id {index} {ret.shape}, dtype {ret.dtype}, device {ret.device}")
        return ret

    def __init__(self, h2d_stream=None, d2h_stream=None, is_a_view_opti=False):
        self._gpu_store = []
        self._offloaded = False
        self._save_event = torch.cuda.Event()
        self._prepare_resume_event = torch.cuda.Event()
        self._resume_event = torch.cuda.Event()
        self._offload_complete_event = torch.cuda.Event()
        self._h2d_stream = h2d_stream
        self._d2h_stream = d2h_stream

        # Datastructures for offload
        self._continuous_cpu_buffer = None
        self._continuous_gpu_buffer = None
        self._offload_tensor_info = []
        self._index_offset = []
        self._index_cpu_buffer = []

        # 优化：使用字典加速 view 查找
        self.is_a_view_opti = is_a_view_opti
        # key: storage_data_ptr, value: list of (index, storage_offset, numel)
        self._storage_index_map = {}

        # 线程锁，用于保护状态变更（特别是异步模式下的状态同步）
        self._state_lock = threading.Lock()

        # 带宽统计相关
        self._offload_start_event = torch.cuda.Event(enable_timing=True)
        self._offload_end_event = torch.cuda.Event(enable_timing=True)
        self._offload_wait_start_event = torch.cuda.Event(enable_timing=True)
        self._offload_wait_end_event = torch.cuda.Event(enable_timing=True)
        self._total_bytes_transferred = 0
        self._resume_start_event = torch.cuda.Event(enable_timing=True)
        self._resume_end_event = torch.cuda.Event(enable_timing=True)
        self._total_bytes_resumed = 0

        # 统计默认流等待H2D完成的时间
        self._resume_wait_start_event = torch.cuda.Event(enable_timing=True)
        self._resume_wait_end_event = torch.cuda.Event(enable_timing=True)
        self._first_resume_tensor_call = True
        self._first_offload_release_call = True

        self._state = ActivationStore.State.NEW
        super().__init__(self._save_tensor, self._resume_tensor)

    def _allocate_cpu_buffers(self):
        if self._continuous_cpu_buffer is not None:
            return
        alignment = 64

        def size_of_tensor(shape, stride):
            id_stride = list(sorted([(i, s) for i, s in enumerate(
                stride) if shape[i] != 1], key=lambda x: x[1]))
            size = 1
            for i, st in id_stride:
                assert size == st, f"stride {stride} size {shape} not continuous"
                size *= shape[i]
            return (size + (alignment - 1)) // alignment * alignment

        self.index_offset = []

        # dtype -> (size, id)
        type_tensors = defaultdict(list)

        for id, (shape, layout, dtype, stride) in enumerate(self._offload_tensor_info):
            assert layout == torch.strided
            # assert dtype == torch.half, f"Only half precision supported, got {dtype} shape {shape}"
            mysize = size_of_tensor(shape, stride)
            type_tensors[dtype].append((mysize, id))

        def nearest_power_of_2(x):
            return 2**(x-1).bit_length()

        def allocate_offset(tensors, max_split=4):
            total_size = sum([x[0] for x in tensors])
            aligned_size = nearest_power_of_2(total_size)
            bin_size = [aligned_size // 2**i for i in range(max_split)]
            bins = [0] * max_split
            tensors = sorted(tensors, key=lambda x: x[0], reverse=True)
            while True:
                bins[-1] += bin_size[-1]
                for i in range(max_split - 1, 0, -1):
                    if bins[i] > bin_size[i]:
                        bins[i] = 0
                        bins[i-1] += bin_size[i-1]

                solution_bins = [x for x in bins if x > 0]
                # print(solution_bins)
                if sum(solution_bins) < total_size:
                    continue
                current_bin = [0] * len(solution_bins)
                # id -> (bin, offset)
                solution = {}
                fit = True
                for size, id in tensors:
                    ok = False
                    for i in range(len(solution_bins)):
                        if current_bin[i] + size <= solution_bins[i]:
                            current_bin[i] += size
                            solution[id] = (i, current_bin[i] - size)
                            ok = True
                            break
                    if not ok:
                        fit = False
                        break
                if fit:
                    assert len(solution) == len(tensors)
                    assert all([x > 0 for x in current_bin])
                    return current_bin, solution

        import psutil
        print(f"rank {torch.distributed.get_rank()} before allocation rss {psutil.Process(os.getpid()).memory_info().rss / 1000000} MB")
        self._continuous_cpu_buffer = {}
        self.index_offset = [None] * len(self._offload_tensor_info)
        for (dtype, tensors) in type_tensors.items():
            from megatron.training import get_args
            if get_args().offload_continuous_buffers:
                bins, solution = allocate_offset(tensors, max_split=8)
            else:
                bins = [t[0] for t in tensors]
                solution = {tensors[i][1]: (i, 0) for i in range(len(tensors))}

            self._continuous_cpu_buffer[dtype] = [
                torch.empty([size], dtype=dtype, pin_memory=True, device='cpu') for size in bins]
            for id, (bin, offset) in solution.items():
                self.index_offset[id] = (bin, offset)
            # print(f"rank {torch.distributed.get_rank()} after allocation {dtype} {bins} elements rss {psutil.Process(os.getpid()).memory_info().rss / 1000000} MB")

        # Print stats
        for dtype, tensors in type_tensors.items():
            total_size = sum([x[0] for x in tensors])
            allocated_size = sum([x.numel() for x in self._continuous_cpu_buffer[dtype]])
            aligned_size = sum([nearest_power_of_2(x.numel())
                               for x in self._continuous_cpu_buffer[dtype]])
            # print(f"rank {torch.distributed.get_rank()} Allocated {allocated_size / 1000000} M elements for {len(tensors)} tensors of type {dtype} total length {total_size} aligned size {aligned_size}")

        for index, (shape, layout, dtype, stride) in enumerate(self._offload_tensor_info):
            bin, offset = self.index_offset[index]
            ctensor = torch.as_strided(
                self._continuous_cpu_buffer[dtype][bin], shape, stride, offset)
            self._index_cpu_buffer.append(ctensor)

    @torch.no_grad()
    # @torch.cuda.nvtx.range("Offload")
    def offload(self, auto_release=False):
        self._change_state(ActivationStore.State.SAVING, ActivationStore.State.OFFLOADED)
        assert not self._offloaded

        self._first_offload_release_call = True
        size = 0
        storage_size = 0
        storages = set()

        with torch.cuda.stream(self._d2h_stream) if self._d2h_stream else contextlib.nullcontext():
            self._save_event.wait()

            self._allocate_cpu_buffers()
            # PairedBarrier.wait_peer()
            # 记录开始时间
            self._offload_start_event.record()
            for index, tensor in enumerate(self._gpu_store):
                buffer = self._index_cpu_buffer[index]
                assert buffer.shape == tensor.shape
                buffer.copy_(tensor, non_blocking=True)
                size += tensor.numel()
                if tensor.storage().data_ptr() not in storages:
                    # print(f"rank {torch.distributed.get_rank()} Storage of tensor {tensor.shape} size {tensor.storage().size()/1000000} MB not in set")
                    storages.add(tensor.storage().data_ptr())
                    storage_size += tensor.storage().nbytes()
                else:
                    # print(f"rank {torch.distributed.get_rank()} Storage of tensor {tensor.shape} size {tensor.storage().size()/1000000} MB already in set")
                    pass
                # print(f"Saving buffer to cpu shape {buffer.shape}, dtype {buffer.dtype}, device {buffer.device}")
                # 剪切操作：每copy完一个tensor就立即释放GPU缓冲区
                # has_nan = torch.any(torch.isnan(tensor))
                # if has_nan:
                #     print("卸载后张量中包含 NaN")
                # self._gpu_store[index] = None
            # PairedBarrier.record()

            # 记录结束时间
            self._offload_end_event.record()
            self._offload_complete_event.record()
        # print(f"rank {torch.distributed.get_rank()} Offloaded {size / 1000000000} Billion elements, {len(self._gpu_store)} tensors, storage size {storage_size / 1000000000} GBytes")

        # 保存数据量用于带宽计算
        self._total_bytes_transferred = storage_size

        self._offloaded = True

    @torch.no_grad()
    # @torch.cuda.nvtx.range("OffloadRelease")
    def offload_release(self):
        self._change_state(ActivationStore.State.OFFLOADED, ActivationStore.State.OFFLOAD_RELEASED)
        assert self._offloaded
        if self._d2h_stream is not None:
            if self._first_offload_release_call:
                self._offload_wait_start_event.record()
            self._offload_complete_event.wait()
            if self._first_offload_release_call:
                self._offload_wait_end_event.record()
                self._offload_wait_end_event.synchronize()
                wait_time_ms = self._offload_wait_start_event.elapsed_time(
                    self._offload_wait_end_event)
                self._offload_end_event.synchronize()
                d2h_time_ms = self._offload_start_event.elapsed_time(self._offload_end_event)
                d2h_time_s = d2h_time_ms / 1000.0
                bandwidth_gbps = (self._total_bytes_transferred / 1e9) / \
                    d2h_time_s if d2h_time_s > 0 else 0
                print(
                    f"rank {torch.distributed.get_rank()} D2H带宽: {bandwidth_gbps:.2f} GB/s "
                    f"({self._total_bytes_transferred / 1e9:.3f} GB / {d2h_time_ms:.2f} ms), "
                    f"默认流等待D2H时间: {wait_time_ms:.2f} ms"
                )
                self._first_offload_release_call = False

        # 记录释放前 GPU store 的信息
        # num_tensors = len(self._gpu_store)
        # if num_tensors > 0 and torch.cuda.is_available():
        #     total_size = sum(t.numel() * t.element_size() for t in self._gpu_store) / (1024**3)
        #     print(f"rank {torch.distributed.get_rank()} [offload_release] "
        #           f"Releasing {num_tensors} GPU tensors, total size: {total_size:.3f} GB")
        #  # 记录执行前的显存

        # # torch.cuda.synchronize()  # 确保之前的操作完成
        # mem_before = torch.cuda.memory_allocated() / (1024**3)  # GB
        # mem_reserved_before = torch.cuda.memory_reserved() / (1024**3)  # GB
        # print(f"rank {torch.distributed.get_rank()} [Before offload_release] "
        #       f"Allocated: {mem_before:.3f} GB, Reserved: {mem_reserved_before:.3f} GB")
        self._gpu_store.clear()
        # 记录执行后的显存

        # torch.cuda.synchronize()  # 确保release操作完成
        # mem_after = torch.cuda.memory_allocated() / (1024**3)  # GB
        # mem_reserved_after = torch.cuda.memory_reserved() / (1024**3)  # GB
        # mem_freed = mem_before - mem_after
        # mem_reserved_freed = mem_reserved_before - mem_reserved_after
        # print(f"rank {torch.distributed.get_rank()} [After offload_release] "
        #       f"Allocated: {mem_after:.3f} GB, Reserved: {mem_reserved_after:.3f} GB")
        # print(f"rank {torch.distributed.get_rank()} [Memory Released] "
        #       f"Freed Allocated: {mem_freed:.3f} GB, Freed Reserved: {mem_reserved_freed:.3f} GB")

    @torch.no_grad()
    # @torch.cuda.nvtx.range("PrepareResume")
    def prepare_resume(self):
        self._change_state(ActivationStore.State.OFFLOAD_RELEASED,
                           ActivationStore.State.RESUME_PREPARED)
        assert self._offloaded
        self._continuous_gpu_buffer = {
            dtype: [torch.empty_like(x, device='cuda') for x in bins] for dtype, bins in self._continuous_cpu_buffer.items()}

        # 计算H2D传输的数据量
        total_bytes = 0
        for dtype, bins in self._continuous_cpu_buffer.items():
            for cpu_buffer in bins:
                total_bytes += cpu_buffer.numel() * cpu_buffer.element_size()
        self._total_bytes_resumed = total_bytes

        for index, (shape, layout, dtype, stride) in enumerate(self._offload_tensor_info):
            bin, offset = self.index_offset[index]
            gtensor = torch.as_strided(
                self._continuous_gpu_buffer[dtype][bin], shape, stride, offset)
            self._gpu_store.append(gtensor)

        self._prepare_resume_event.record()

    @torch.no_grad()
    # @torch.cuda.nvtx.range("Resume")
    def resume(self):
        self._change_state(ActivationStore.State.RESUME_PREPARED, ActivationStore.State.RESUMED)
        assert self._offloaded

        # 重置标志位，准备统计本次resume的等待时间
        self._first_resume_tensor_call = True

        original_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self._h2d_stream) if self._h2d_stream else contextlib.nullcontext():
            self._prepare_resume_event.wait()
            self._offload_complete_event.wait()

            # 记录H2D开始时间
            self._resume_start_event.record()

            # PairedBarrier.wait_peer()
            for dtype, bins in self._continuous_cpu_buffer.items():
                for (cpu, gpu) in zip(bins, self._continuous_gpu_buffer[dtype]):
                    gpu.copy_(cpu, non_blocking=True)
            # PairedBarrier.record()

            # 记录H2D结束时间
            self._resume_end_event.record()
            self._resume_event.record()
            # for dtype, bins in self._continuous_cpu_buffer.items():
            #     for (cpu, gpu) in zip(bins, self._continuous_gpu_buffer[dtype]):
            #         has_nan = torch.any(torch.isnan(gpu))
            #         if has_nan:
            #             print("resume后张量中包含 NaN")

        self._offloaded = False

    @torch.no_grad()
    # @torch.cuda.nvtx.range("ResumeRelease")
    def resume_release(self):
        self._change_state(ActivationStore.State.RESUME_USED, ActivationStore.State.RESUME_RELEASED)
        assert all([x is None for x in self._gpu_store])
        assert all([all([x is None for x in y]) for y in self._continuous_gpu_buffer.values()])
        self._resume_event.wait()

        self._gpu_store.clear()
        self._continuous_gpu_buffer.clear()
        self._storage_index_map.clear()  # 清空索引字典

    def reset_state(self):
        self._change_state(ActivationStore.State.RESUME_RELEASED, ActivationStore.State.NEW)


offload_stream = None
d2h_stream = None


def get_offload_h2d_stream():
    global offload_stream
    if offload_stream is None:
        offload_stream = torch.cuda.Stream()
    return offload_stream


def get_offload_d2h_stream():
    global d2h_stream
    if d2h_stream is None:
        d2h_stream = torch.cuda.Stream()
    return d2h_stream

# We expect the calling order for every store to be:
# get_for_offload
# offload
# offload_release
# prepare_resume
# resume
# resume_release


class ActivationStorePool:
    def __init__(self) -> None:
        self._pool = []
        self._stage_queues = [[] for x in range(6)]

    def get_for_offload(self, is_a_view_opti=False) -> ActivationStore:
        if self._pool:
            ret = self._pool.pop(-1)
            ret.reset_state()
        else:
            ret = ActivationStore(get_offload_h2d_stream(),
                                  get_offload_d2h_stream(), is_a_view_opti=is_a_view_opti)
        self._current_store = ret
        self._stage_queues[0].append(ret)
        return ret

    def pop_call_push(self, stage_idx, func):
        assert self._stage_queues[stage_idx], f"stage_idx {stage_idx} is empty"
        store = self._stage_queues[stage_idx].pop(0)
        ret = func(store)
        self._stage_queues[stage_idx + 1].append(store)
        return ret

    def offload(self, auto_release=False):
        return self.pop_call_push(0, lambda x: x.offload(auto_release=auto_release))

    def offload_release(self):
        return self.pop_call_push(1, lambda x: x.offload_release())

    def prepare_resume(self):
        return self.pop_call_push(2, lambda x: x.prepare_resume())

    def resume(self):
        return self.pop_call_push(3, lambda x: x.resume())

    def resume_release(self, store_deprecated=None):
        self.pop_call_push(4, lambda x: x.resume_release())
        self._pool.append(self._stage_queues[5].pop(0))

    def is_empty(self):
        return sum([len(x) for x in self._stage_queues]) == 0
