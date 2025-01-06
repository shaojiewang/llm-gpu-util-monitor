from pynvml import *
import time
import argparse
import torch

from pth_gemm_test import timer_matmul

def nvml_tc_utils(func):

    def wrapper(*args, **kwargs):
        nvmlInit()
        device0 = nvmlDeviceGetHandleByIndex(0)
        is_gpm_supported = nvmlGpmQueryDeviceSupport(device0)
        sample_before_function = nvmlGpmSampleAlloc()
        sample_before_function_device0 = nvmlGpmSampleGet(device0, sample_before_function)
        
        result = func(*args, **kwargs)

        sample_after_function = nvmlGpmSampleAlloc()
        sample_after_function_device0 = nvmlGpmSampleGet(device0, sample_after_function)

        gpm_metrics = c_nvmlGpmMetricsGet_t()
        gpm_metrics.version = NVML_GPM_METRICS_GET_VERSION
        gpm_metrics.numMetrics = NVML_GPM_METRIC_MAX
        gpm_metrics.sample1 = sample_before_function_device0
        gpm_metrics.sample2 = sample_after_function_device0

        for id in range(1, NVML_GPM_METRIC_MAX):
            gpm_metrics.metrics[id - 1].metricId = id

        gpm_metrics = nvmlGpmMetricsGet(gpm_metrics)

        for i in range(NVML_GPM_METRIC_MAX):
            print(gpm_metrics.metrics[i])

        print(f"any  tensor util = {gpm_metrics.metrics[NVML_GPM_METRIC_ANY_TENSOR_UTIL-1]}")
        print(f"hmma tensor util = {gpm_metrics.metrics[NVML_GPM_METRIC_HMMA_TENSOR_UTIL-1]}")

        nvmlGpmSampleFree(sample_before_function_device0)
        nvmlGpmSampleFree(sample_after_function_device0)

        nvmlShutdown()
        return result

    return wrapper

def cuda_event_timer(func):

    def wrapper(*args, **kwargs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        result = func(*args, **kwargs)

        end_event.record()
        torch.cuda.synchronize()

        elapsed_time_ms = start_event.elapsed_time(end_event)
        return elapsed_time_ms

    return wrapper

@nvml_tc_utils
@cuda_event_timer
def torch_matmul(a, b, total_num):
    for i in range(total_num):
        c = a @ b
    return c


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--m", type=int)
    parser.add_argument("-n", "--n", type=int)
    parser.add_argument("-k", "--k", type=int)
    parser.add_argument("-t_num", "--total_num", type=int)
    args = parser.parse_args()

    nvmlInit()
    device=nvmlDeviceGetHandleByIndex(0)
    a=nvmlGpmQueryDeviceSupport(device)

    dt = torch.bfloat16
    m = args.m
    n = args.n
    k = args.k

    total_num = args.total_num

    a_full = torch.randn(m, k, dtype=dt, device='cuda')
    b_full = torch.randn(k, n, dtype=dt, device='cuda')
    ab_full = a_full @ b_full

    elapsed_time_ms = torch_matmul(a_full, b_full, total_num)

    elapsed_time_ms_avg = elapsed_time_ms / total_num
    total_tflops = m * n * k * 2 / 1000 / 1000 / 1000
    total_GBs = (a_full.nbytes + b_full.nbytes + ab_full.nbytes) / 1000 / 1000
    str_time = """dt={F_dt}, mnk={F_m}, {F_n}, {F_k}, elapsed_time_ms={F_time}, tflops={F_tflops}, GB/s={F_bw}, real_tc_util={F_tc_real_util}%"""
    tflops = total_tflops / elapsed_time_ms_avg
    hbm_bw = total_GBs / elapsed_time_ms_avg
    tc_real_util = tflops / 990 * 100
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
    print('cuda event timer')
    print(str_time.format(F_dt=dt_str, F_m=m, F_n=n, F_k=k, F_time=round(elapsed_time_ms_avg, 2), F_tflops=round(tflops, 2), F_bw=round(hbm_bw, 2), F_tc_real_util=round(tc_real_util, 2)))

