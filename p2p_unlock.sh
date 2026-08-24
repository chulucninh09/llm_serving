#!/bin/bash
# open-gpu / P2P references:
# https://github.com/aikitoria/open-gpu-kernel-modules/
# https://smcleod.net/2026/02/patching-nvidias-driver-and-vllm-to-enable-p2p-on-consumer-gpus/
#
# -----------------------------------------------------------------------------
# NVIDIA + Secure Boot: modprobe fails with "Key was rejected by service" and
# nvidia-smi says it could not communicate with the driver — the kernel rejects
# the module signer until your key is enrolled in MOK (Machine Owner Key).
#
# Keys generated on this machine (keep MOK.key.pem private; chmod 600):
#   /root/module-mok-keys/MOK.key.pem   signing private key
#   /root/module-mok-keys/MOK.crt.pem   cert PEM
#   /root/module-mok-keys/MOK.der       cert DER for mokutil --import
# Helper script: /root/module-mok-keys/enroll-mok-import.sh
#
# 1) One-time enrollment (interactive — pick a password and remember it):
#      sudo mokutil --import /root/module-mok-keys/MOK.der
#      # or: sudo /root/module-mok-keys/enroll-mok-import.sh
#
#    Optional: sudo mokutil --list-new   (confirm pending import before reboot)
#
# 2) Reboot. At the Shim / blue MOK Manager screen, choose Enroll MOK / Continue
#    and confirm with the SAME password mokutil asked for.
#
# 3) After Linux boots, kernel accepts modules signed by that key. Verify:
#      sudo modprobe nvidia nvidia-uvm nvidia-modeset
#      nvidia-smi
#
# Kernel uses sha512 module signatures — sign after every (re)install of the .ko
# files under drivers/video/. Example after rebuilding open-gpu / copying .ko:
#      KVER=$(uname -r)
#      HDR=/usr/src/linux-headers-$KVER
#      KEYDIR=/root/module-mok-keys
#      for ko in /lib/modules/$KVER/kernel/drivers/video/nvidia*.ko; do
#        "$HDR/scripts/sign-file" sha512 "$KEYDIR/MOK.key.pem" "$KEYDIR/MOK.crt.pem" "$ko"
#      done
#      sudo depmod -a
#
# If you prefer not to use MOK: disable Secure Boot in UEFI firmware instead,
# then modprobe normally (no signing needed for plain external modules on many
# setups — YMMV with lockdown/policy).
#
# Still broken? Check: sudo dmesg | tail -80
# -----------------------------------------------------------------------------

exit 0
