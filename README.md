# GPU Tensor Core Utilization Monitor to Nvidia

## how to use
```
python tensor_core_monitor.py -m 4096 -n 4096 -k 8192 -t_num 10000
```
Then I got:
```
any  tensor util = c_nvmlGpmMetric_t(metricId: 5, nvmlReturn: 0, value: 81.90776987469786, metricInfo: <pynvml.c_metricInfo_t object at 0x7f2550e76ac0>)
hmma tensor util = c_nvmlGpmMetric_t(metricId: 7, nvmlReturn: 0, value: 81.90776858965575, metricInfo: <pynvml.c_metricInfo_t object at 0x7f2550e76ac0>)
cuda event timer
dt=bfloat16, mnk=4096, 4096, 8192, elapsed_time_ms=0.45, tflops=616.1, GB/s=376.03, real_tc_util=62.23%
```
There is a difference between 62% and 82%. Could you tell us the reason that GPM gives about 30% higher than the real time test?

When I use 4k matmul case, still got a difference between monitor and the real-time test.

```
python tensor_core_monitor.py -m 4096 -n 4096 -k 4096 -t_num 10000
```

I got:

```
any  tensor util = c_nvmlGpmMetric_t(metricId: 5, nvmlReturn: 0, value: 75.32818099823899, metricInfo: <pynvml.c_metricInfo_t object at 0x7fe8a14daac0>)
hmma tensor util = c_nvmlGpmMetric_t(metricId: 7, nvmlReturn: 0, value: 75.32817929716414, metricInfo: <pynvml.c_metricInfo_t object at 0x7fe8a14daac0>)
cuda event timer
dt=bfloat16, mnk=4096, 4096, 4096, elapsed_time_ms=0.22, tflops=623.76, GB/s=456.85, real_tc_util=63.01%
```

