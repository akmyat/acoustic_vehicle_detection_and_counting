# Install Picamera2
sudo apt install -y python3-picamera2
sudo apt install -y python3-pyqt5 python3-opengl

# Increase CMA memory for camera system
echo "gpu_mem=256" >> /boot/config.txt

# Config for DKMS preview window
echo "dtoverlay=vc4-kms-v3d" >> /boot/config.txt

# Enable Glamor driver
echo "sudo raspi-config > Advanced Options > Enable Glamor graphic acceleration > reboot"

# Config for camear v1.3 driver
echo "dtoverlay=ov5647" >> /boot/config.txt
