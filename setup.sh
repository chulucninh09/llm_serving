# Setup tailscale for easier access
curl -fsSL https://tailscale.com/install.sh | sh
tailscale up --login-server=https://headscale.fidt.vn --reset --accept-dns --accept-routes

# Download nessessary packages

# Update ubuntu mirror
sudo sed -i 's/archive.ubuntu.com/mirror.azvps.vn\/ubuntu/g' /etc/apt/sources.list.d/ubuntu.list

# Install gh cli
(type -p wget >/dev/null || (sudo apt update && sudo apt install wget -y)) \
	&& sudo mkdir -p -m 755 /etc/apt/keyrings \
	&& out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
	&& cat $out | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
	&& sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
	&& sudo mkdir -p -m 755 /etc/apt/sources.list.d \
	&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
	&& sudo apt update \
	&& sudo apt install gh -y

# Install ubuntu-drivers, cmake, ccache, nvtop, xorg, nvidia-settings
apt install ubuntu-drivers-common cmake ccache nvtop \
 xorg nvidia-settings build-essential libomp-dev libssl-dev \
 numactl openssl ffmpeg tmux glances python3.12-dev -y

# Install nvidia-driver and nvidia-smi
# RTX 5090 (Blackwell) requires the open kernel modules
apt install -y linux-headers-$(uname -r)
ubuntu-drivers install nvidia-driver-580-open --gpgpu

# Run gpu_undervolt.sh
# ./gpu_undervolt.sh

# install cuda toolkit
source ./install_cuda.sh

# Mount data drives (reproducible from a re-provisioned box):
#   vdb = HF weights  -> ext4 (kept), tuned for sequential read
#   vdc = KV offload  -> XFS, tuned for random whole-file 512KB reads
# nofail so boot succeeds even if the disk is absent.
#
# vdc: XFS (random-IO profile)
# mkfs.xfs -f -L llm-kv -m reflink=0 -d agcount=16 -l internal,size=256m /dev/vdc
# mkdir -p /mnt/llm-data/kv-cache
# echo "LABEL=llm-kv /mnt/llm-data/kv-cache xfs defaults,noatime,nofail,x-systemd.device-timeout=10s,logbsize=256k 0 0" >> /etc/fstab
# mkdir -p /mnt/llm-data/kv-cache/sglang
# xfs_io -c 'extsize 512k' /mnt/llm-data/kv-cache
# xfs_io -c 'extsize 512k' /mnt/llm-data/kv-cache/sglang
#
# vdb: ext4 (sequential-IO profile), keep existing filesystem
# uuid=$(blkid -s UUID -o value /dev/vdb)
# mkdir -p /mnt/llm-data/huggingface
# echo "UUID=$uuid /mnt/llm-data/huggingface ext4 defaults,noatime,nofail,x-systemd.device-timeout=10s 0 2" >> /etc/fstab
# mount -a
#
# Per-disk block-layer tuning (persistent) -> /etc/udev/rules.d/99-llm-data.rules:
#   ACTION=="add|change", KERNEL=="vdb", SUBSYSTEM=="block", ATTR{queue/scheduler}="none", ATTR{queue/read_ahead_kb}="16384", ATTR{queue/max_sectors_kb}="4096"
#   ACTION=="add|change", KERNEL=="vdc", SUBSYSTEM=="block", ATTR{queue/scheduler}="none", ATTR{queue/read_ahead_kb}="128", ATTR{queue/max_sectors_kb}="1024"
#
# Writeback tuning -> /etc/sysctl.d/99-llm-serving.conf:
#   vm.dirty_background_ratio=5
#   vm.dirty_ratio=10
#   vm.dirty_expire_centisecs=2000
#
# systemctl enable --now fstrim.timer   (weekly trim; no mount-time discard)

# Point Hugging Face cache at the data drive
grep -q 'export HF_HOME="/mnt/llm-data/huggingface"' ~/.bashrc \
  || printf '\n# Hugging Face cache on data drive\nexport HF_HOME="/mnt/llm-data/huggingface"\n' >> ~/.bashrc

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh


# # To use llama.cpp
# # Install blis
# cd blis
# ./configure --enable-cblas -t openmp,pthreads auto
# # will install to /usr/local/ by default.
# make -j
# sudo make install

# # Compile llama.cpp
# ./build_llama.cpp.sh

# # Run llama.cpp
# ./run_llama.cpp.sh

# # To use vLLM
# # Install docker
# ./install_docker.sh

# # Install nvidia container toolkit
# ./install_nvidia_container_toolkit.sh

# # Run vLLM
# ./run_vllm.sh