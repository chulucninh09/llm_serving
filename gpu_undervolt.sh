# Generate basic config
nvidia-xconfig

# Start x server
X :0 &

# Add coolbits
nano /etc/X11/xorg.conf
Section "Device"
    Identifier     "Device1"
    Driver         "nvidia"
    VendorName     "NVIDIA Corporation"
    BusID          "PCI:X:X:X"    # nvidia-smi will show this
    Option         "Coolbits" "28"
    Option         "AllowEmptyInitialConfiguration" "True"
EndSection

# Set compute mode
nvidia-smi --gom=1

# Limit clock
nvidia-smi -pm 1
nvidia-smi -lgc 0,1100
nvidia-smi -lmc 0,7000
nvidia-smi -pl 200


# Overclock for undervolt
export DISPLAY=:0
nvidia-settings -a [gpu:0]/GPUGraphicsClockOffsetAllPerformanceLevels=220
nvidia-settings -a [gpu:1]/GPUGraphicsClockOffsetAllPerformanceLevels=220

nvidia-settings -a [gpu:0]/GPUMemoryTransferRateOffsetAllPerformanceLevels=700
nvidia-settings -a [gpu:1]/GPUMemoryTransferRateOffsetAllPerformanceLevels=700


watch -n0.5 "nvidia-smi -i 0 -q -d POWER,CLOCK"
watch -n0.5 "nvidia-smi -i 1 -q -d POWER,CLOCK"