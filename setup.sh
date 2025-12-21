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
apt install ubuntu-drivers-common cmake ccache nvtop xorg nvidia-settings build-essential libomp-dev -y

# Install nvidia-driver and nvidia-smi
ubuntu-drivers install nvidia:580-server --gpgpu
apt install -y nvidia-utils-580-server

# Run gpu_undervolt.sh
./gpu_undervolt.sh

# install cuda toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-1
export PATH=/usr/local/cuda/bin:$PATH

# Mount drive
mkdir -p /mnt/llm-data/huggingface
/dev/vdb /mnt/llm-data/huggingface ext4
systemctl daemon-reload
mount -a

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Login to huggingface
uvx hf auth login

# Download model
uvx hf download 

# Install blis
cd blis
./configure --enable-cblas -t openmp,pthreads auto
# will install to /usr/local/ by default.
make -j
sudo make install

# Compile llama.cpp
./build_llama.cpp.sh