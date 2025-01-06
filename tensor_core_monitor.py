from pynvml import *
import time
import argparse
import torch

from pth_gemm_test import timer_matmul

if __name__ == "__main__":
    nvmlInit()
    device=nvmlDeviceGetHandleByIndex(0)
    a=nvmlGpmQueryDeviceSupport(device)

    dt = torch.bfloat16
    m = 4*1024
    n = 4*1024
    k = 4*1024

    total_num = 10000

    a_full = torch.randn(m, k, dtype=dt, device='cuda')
    b_full = torch.randn(k, n, dtype=dt, device='cuda')
    ab_full = a_full @ b_full

    b0=nvmlGpmSampleAlloc()
    b0_get=nvmlGpmSampleGet(device, b0)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    for i in range(total_num):
        c= a_full @ b_full

    end_event.record()
    torch.cuda.synchronize()

    b1=nvmlGpmSampleAlloc()
    b1_get=nvmlGpmSampleGet(device, b1)

    mg=c_nvmlGpmMetricsGet_t()

    mg.version = NVML_GPM_METRICS_GET_VERSION
    mg.numMetrics = NVML_GPM_METRIC_MAX
    mg.sample1 = b0_get
    mg.sample2 = b1_get

    for id in range(1, NVML_GPM_METRIC_MAX):
        mg.metrics[id - 1].metricId = id

    mg = nvmlGpmMetricsGet(mg)

    # print(mg)
    for i in range(NVML_GPM_METRIC_MAX):
        print(mg.metrics[i])
    print(f"tensor util={mg.metrics[NVML_GPM_METRIC_ANY_TENSOR_UTIL-1]}")
    print(f"fp16 tensor util={mg.metrics[NVML_GPM_METRIC_HMMA_TENSOR_UTIL-1]}")

    print('cuda event timer')
    elapsed_time_ms = start_event.elapsed_time(end_event)
    elapsed_time_ms_avg = elapsed_time_ms / total_num
    total_tflops = m * n * k * 2 / 1000 / 1000 / 1000
    total_GBs = (a_full.nbytes + b_full.nbytes + c.nbytes) / 1000 / 1000
    str_time = """dt={F_dt}, mnk={F_m}, {F_n}, {F_k}, elapsed_time_ms={F_time}, tflops={F_tflops}, GB/s={F_bw}"""
    tflops = total_tflops / elapsed_time_ms_avg
    hbm_bw = total_GBs / elapsed_time_ms_avg
    if dt == torch.double:
        dt_str = "double"
    elif dt == torch.float:
        dt_str = "float32"
    elif dt == torch.half:
        dt_str = "float16"
    elif dt == torch.bfloat16:
        dt_str = "bfloat16"
    elif dt == torch.float8:
        dt_str = "float8"
    print(str_time.format(F_dt=dt_str, F_m=m, F_n=n, F_k=k, F_time=round(elapsed_time_ms_avg, 2), F_tflops=round(tflops, 2), F_bw=round(hbm_bw, 2)))


    nvmlGpmSampleFree(b0_get)
    nvmlGpmSampleFree(b1_get)

    nvmlShutdown()

