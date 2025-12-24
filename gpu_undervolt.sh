# Generate basic config
export DISPLAY=:0
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
nvidia-smi -lgc 0,1450
nvidia-smi -lmc 0,11000
nvidia-smi -pl 200


# Overclock for undervolt
nvidia-settings -a [gpu:0]/GPUGraphicsClockOffsetAllPerformanceLevels=220
nvidia-settings -a [gpu:1]/GPUGraphicsClockOffsetAllPerformanceLevels=220

nvidia-settings -a [gpu:0]/GPUMemoryTransferRateOffsetAllPerformanceLevels=1300
nvidia-settings -a [gpu:1]/GPUMemoryTransferRateOffsetAllPerformanceLevels=1300


watch -n0.5 "nvidia-smi -i 0 -q -d POWER,CLOCK"
watch -n0.5 "nvidia-smi -i 1 -q -d POWER,CLOCK"