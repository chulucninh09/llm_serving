
# Start x server
X :0 &

# Generate basic config
export DISPLAY=:0
nvidia-xconfig

# Add coolbits
# Section "Device"
#     Identifier     "Device0"
#     Driver         "nvidia"
#     VendorName     "NVIDIA Corporation"
#     BusID          "PCI:X:X:X"    # nvidia-smi will show this
#     Option         "Coolbits" "28"
#     Option         "AllowEmptyInitialConfiguration" "True"
# EndSection

# Overclock for undervolt
nvidia-settings -a [gpu:0]/GPUGraphicsClockOffsetAllPerformanceLevels=240

# Limit clock
nvidia-smi -pm 1
nvidia-smi -lgc 210,1700
nvidia-smi -pl 200