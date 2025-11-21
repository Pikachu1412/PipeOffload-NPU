#!/usr/bin/env python3
"""
示例：使用 saved_tensors_hooks 捕获 dropout 操作中的张量
"""

from contextlib import contextmanager
import time
import torch
import torch_npu
import torch.nn.functional as F
@contextmanager
def record_memory_delta_and_time(event_name: str = "event", device = None):
    # yield
    # return
    # if torch.distributed.get_rank() != 0:
    #     yield
    #     return
    device = torch_npu.npu.current_device()
    # 查询指定NPU设备的已分配内存
    start_allocated = torch_npu.npu.memory_allocated(device)
    # device = torch.cuda.current_device() if device is None else device
    # torch.cuda.synchronize(device)
    # start_allocated = torch.cuda.memory_allocated(device)
    start_time = time.perf_counter()
    try:
        yield
    finally:
        wait_time = (time.perf_counter() - start_time) * 1000
        # torch.cuda.synchronize(device)
        end_allocated = torch_npu.npu.memory_allocated(device)
        # end_allocated = torch.cuda.memory_allocated(device)
        # end_reserved = torch.cuda.memory_reserved(device)
        delta_allocated = (end_allocated - start_allocated) / (1024 * 1024)
        print(f"{event_name} wait time: {wait_time:.4f} ms start: {start_allocated / (1024 * 1024):.3f} MB end: {end_allocated / (1024 * 1024):.3f} MB allocated Δ: {delta_allocated:.3f} MB")

class DropoutTensorHook:
    """用于捕获 dropout 操作中保存的张量的钩子类"""
    
    def __init__(self):
        self.saved_tensors = []
        self.pack_hook_called = False
        self.unpack_hook_called = False
    
    def pack_hook(self, x):
        """在保存张量时调用的钩子"""
        print(f"pack_hook 被调用，捕获张量:")
        print(f"  形状: {x.shape}")
        print(f"  数据类型: {x.dtype}")
        print(f"  设备: {x.device}")
        print(f"  是否需要梯度: {x.requires_grad}")
        print("-" * 50)
        print(f"in hook ptr is {x.storage().data_ptr()}")
        self.saved_tensors.append(x)
        self.pack_hook_called = True
        return x

    def unpack_hook(self, x):
        """在恢复张量时调用的钩子"""
        print(f"unpack_hook 被调用，恢复张量:")
        print(f"  形状: {x.shape}")
        print(f"  数据类型: {x.dtype}")
        print(f"  设备: {x.device}")
        print("-" * 50)
        self.unpack_hook_called = True
        return x

def capture_dropout_with_hooks():
    """使用 saved_tensors_hooks 捕获 dropout 操作中的张量"""
    print("=" * 60)
    print("开始捕获 dropout 操作中的张量")
    print("=" * 60)
    
    # 创建输入张量
    x = torch.randn(4096, 1, 1024, requires_grad=True).npu()
    x1 = torch.randn(4096, 1024, 1, requires_grad=True).npu()
    bias = torch.randn(4096 , 1 ,1024, requires_grad=True).npu()
    res = torch.randn(4096 , 1 ,1024, requires_grad=True).npu()
    print(f"输入张量 x: {x.shape}, requires_grad={x.requires_grad}")
    
    # 创建钩子实例
    hook = DropoutTensorHook()
    
    # 设置 saved_tensors_hooks
    with torch.autograd.graph.saved_tensors_hooks(hook.pack_hook, hook.unpack_hook):
        print("\n执行 dropout 操作...")
        prob = 0.2
        training = True
        with record_memory_delta_and_time("123"):
            # x = x + bias
            print(f"before ptr is {x.storage().data_ptr()}")
            out = x @ x1
            # out = torch.nn.functional.dropout(x, p=prob, training=training)
            # out = res + out
        # out = F.dropout(x, p=prob, training=training)
        print(f"dropout 输出: {out.shape}")
        
        # 创建一个简单的计算图
        loss = out.sum()
        print(f"计算损失: {loss.item()}")
        
        # 反向传播来触发钩子
        print("\n执行反向传播...")
        loss.backward()
    
    print("\n" + "=" * 60)
    print("钩子调用统计:")
    print(f"  pack_hook 被调用: {hook.pack_hook_called}")
    print(f"  unpack_hook 被调用: {hook.unpack_hook_called}")
    print(f"  总共捕获张量数量: {len(hook.saved_tensors)}")
    print("=" * 60)
    
    # 显示捕获的张量详细信息
    if hook.saved_tensors:
        print("\n捕获的张量详细信息:")
        for i, tensor in enumerate(hook.saved_tensors):
            print(f"  张量 {i+1}:")
            print(f"    形状: {tensor.shape}")
            print(f"    数据类型: {tensor.dtype}")
            print(f"    设备: {tensor.device}")
            print(f"    是否需要梯度: {tensor.requires_grad}")
            print(f"    是否被修改: {torch.any(tensor != x)}")
    
    return hook.saved_tensors


if __name__ == "__main__":
    # 捕获 dropout 操作中的张量
    captured_tensors = capture_dropout_with_hooks()
    