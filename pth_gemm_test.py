import torch

def timer_matmul(dt = torch.double, m = 1024, n = 1024, k = 1024, warmup_num = 2, total_num = 10):
    cuda0 = torch.device('cuda:0')
    a_full = torch.randn(m, k, dtype=dt, device='cuda')
    b_full = torch.randn(k, n, dtype=dt, device='cuda')

    for i in range(warmup_num):
        ab_full = a_full @ b_full

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    for i in range(total_num):
        c = a_full @ b_full
    
    end_event.record()

    torch.cuda.synchronize()
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


