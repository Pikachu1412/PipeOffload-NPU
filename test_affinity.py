#!/usr/bin/env python3
"""
快速 CPU affinity 测试脚本
"""
import os

print("快速 CPU Affinity 检查")
print("-" * 50)

# 1. 总核心数
total_cpus = os.cpu_count()
print(f"系统总 CPU 核心数: {total_cpus}")

# 2. 当前可用核心
try:
    current_affinity = os.sched_getaffinity(0)
    available_cores = sorted(current_affinity)
    print(f"当前进程可用核心: {available_cores}")
    print(f"可用核心数量: {len(available_cores)}")
    
    if len(available_cores) == 0:
        print("\n❌ 问题: 没有可用的 CPU 核心!")
        print("   这会导致 GPUAffinityError")
    elif len(available_cores) < total_cpus:
        print(f"\n⚠️  警告: CPU affinity 受限!")
        print(f"   只能使用 {len(available_cores)}/{total_cpus} 个核心")
        print(f"   限制比例: {len(available_cores)/total_cpus*100:.1f}%")
    else:
        print("\n✅ 正常: 所有 CPU 核心都可用")
        
except Exception as e:
    print(f"\n❌ 错误: 无法获取 affinity - {e}")

# 3. 测试 GPU 和 NVML
print("\n" + "-" * 50)
print("GPU 和 NVML 检查")
print("-" * 50)

try:
    import torch
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ 检测到 {gpu_count} 个 GPU")
        
        try:
            import pynvml
            pynvml.nvmlInit()
            
            for i in range(min(gpu_count, 2)):  # 只测试前2个GPU
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # 获取推荐核心
                nvml_affinity_elements = (total_cpus + 63) // 64
                node_affinity_raw = pynvml.nvmlDeviceGetCpuAffinityWithinScope(
                    handle, nvml_affinity_elements, pynvml.NVML_AFFINITY_SCOPE_NODE
                )
                
                affinity_string = ''
                for j in node_affinity_raw:
                    affinity_string = '{:064b}'.format(j) + affinity_string
                affinity_list = [int(x) for x in affinity_string]
                affinity_list.reverse()
                recommended_cores = [idx for idx, e in enumerate(affinity_list) if e != 0]
                
                # 计算交集
                intersection = sorted(list(set(recommended_cores) & current_affinity))
                
                print(f"\nGPU {i}:")
                print(f"  NVML 推荐核心数: {len(recommended_cores)}")
                print(f"  与可用核心交集: {len(intersection)} 个")
                
                if len(intersection) == 0:
                    print(f"  ❌ 没有交集! 这会导致错误!")
                elif len(intersection) < len(recommended_cores):
                    print(f"  ⚠️  交集受限 ({len(intersection)}/{len(recommended_cores)})")
                else:
                    print(f"  ✅ 正常")
            
            pynvml.nvmlShutdown()
            
        except ImportError:
            print("⚠️  pynvml 未安装,无法测试 GPU affinity")
        except Exception as e:
            print(f"❌ NVML 测试失败: {e}")
    else:
        print("❌ CUDA 不可用")
except ImportError:
    print("❌ PyTorch 未安装")

print("\n" + "=" * 50)
print("结论:")
print("=" * 50)
if len(available_cores) == 0:
    print("❌ CPU affinity 被完全限制,需要修复!")
    print("   解决方案: 使用已修改的代码(添加了异常处理)")
elif len(available_cores) < total_cpus:
    print("⚠️  CPU affinity 受限,可能导致问题")
    print("   建议: 检查容器/SLURM 配置,或使用修改后的代码")
else:
    print("✅ CPU affinity 正常")
