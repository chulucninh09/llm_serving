# Disable power limit for mainboard
ipmitool dcmi power deactivate
ipmitool dcmi power get_limit

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

# Overclock for undervolt
export DISPLAY=:0
nvidia-settings -a [gpu:0]/GPUGraphicsClockOffsetAllPerformanceLevels=200
nvidia-settings -a [gpu:1]/GPUGraphicsClockOffsetAllPerformanceLevels=230

# Memory uses fixed voltage => underclock to lower power consumption
nvidia-settings -a [gpu:0]/GPUMemoryTransferRateOffsetAllPerformanceLevels=-2000
nvidia-settings -a [gpu:1]/GPUMemoryTransferRateOffsetAllPerformanceLevels=-2000

# Limit clock
nvidia-smi -pm 1
nvidia-smi --gom=1
nvidia-smi -lgc 0,1600
nvidia-smi -lmc 0,8501
nvidia-smi -pl 250

# Monitor script
watch -n0.5 "nvidia-smi -i 0 -q -d POWER,CLOCK"
watch -n0.5 "nvidia-smi -i 1 -q -d POWER,CLOCK"