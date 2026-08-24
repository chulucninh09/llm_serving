import pynvml

pynvml.nvmlInit()

device_count = pynvml.nvmlDeviceGetCount()
print(f"Found {device_count} GPUs\n")

handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(device_count)]

for i, handle in enumerate(handles):
    name = pynvml.nvmlDeviceGetName(handle)
    print(f"GPU {i}: {name}")

print()

caps = {
    "NVLINK":   pynvml.NVML_P2P_CAPS_INDEX_NVLINK,
    "ATOMICS":  pynvml.NVML_P2P_CAPS_INDEX_ATOMICS,
    "PROP":     pynvml.NVML_P2P_CAPS_INDEX_PROP,
}

status_names = {
    pynvml.NVML_P2P_STATUS_OK:             "OK ✅",
    pynvml.NVML_P2P_STATUS_GPU_NOT_SUPPORTED:      "GPU_NOT_SUPPORTED ❌",
    pynvml.NVML_P2P_STATUS_IOH_TOPOLOGY_NOT_SUPPORTED: "IOH_TOPOLOGY_NOT_SUPPORTED ❌",
    pynvml.NVML_P2P_STATUS_DISABLED_BY_REGKEY:     "DISABLED_BY_REGKEY ❌",
    pynvml.NVML_P2P_STATUS_NOT_SUPPORTED:          "NOT_SUPPORTED ❌",
    pynvml.NVML_P2P_STATUS_UNKNOWN:                "UNKNOWN ❌",
}

for i in range(device_count):
    for j in range(device_count):
        if i >= j:
            continue
        print(f"GPU {i} <-> GPU {j}:")
        for cap_name, cap_index in caps.items():
            try:
                status = pynvml.nvmlDeviceGetP2PStatus(handles[i], handles[j], cap_index)
                print(f"  {cap_name:10s}: {status_names.get(status, f'UNKNOWN({status})')}")
            except pynvml.NVMLError as e:
                print(f"  {cap_name:10s}: ERROR - {e}")
        print()

pynvml.nvmlShutdown()