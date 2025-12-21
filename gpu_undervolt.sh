

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

# Overclock for undervolt
nvidia-settings -a [gpu:0]/GPUGraphicsClockOffsetAllPerformanceLevels=220
nvidia-settings -a [gpu:1]/GPUGraphicsClockOffsetAllPerformanceLevels=220

# Limit clock
nvidia-smi -pm 1
nvidia-smi -lgc 210,1830
nvidia-smi -pl 200