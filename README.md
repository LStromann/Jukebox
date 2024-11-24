# OpenAI Jukebox Server Setup

This document outlines the steps to set up and troubleshoot OpenAI Jukebox on an Ubuntu 22.04 server with NVIDIA GPU.

---

## 1. Operating System
- **System**: Ubuntu 22.04 LTS
- **Steps Taken**:
  - Confirmed OS compatibility for NVIDIA drivers and CUDA using:
    ```bash
    lsb_release -a
    ```
  - No changes needed.

---

## 2. NVIDIA Driver
- **Installed Driver**: `535.183.01`
- **Steps Taken**:
  - Verified the driver compatibility with `nvidia-smi`. Output confirmed CUDA runtime version compatibility (`12.2` supported, fine for CUDA 11.1):
    ```bash
    nvidia-smi
    ```
  - Skipped driver installation during CUDA Toolkit setup as the current driver was newer and compatible.

---

## 3. CUDA Toolkit
- **Installed Version**: CUDA 11.1
- **Steps Taken**:
  1. Downloaded and installed CUDA 11.1 (Ubuntu 20.04 version) because Ubuntu 22.04 is not officially supported for this version:
     ```bash
     wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
     sudo sh cuda_11.1.1_455.32.00_linux.run
     ```
  2. Selected "Install Toolkit Only" and skipped driver installation.
  3. Added CUDA paths to `.bashrc`:
     ```bash
     echo 'export PATH=/usr/local/cuda-11.1/bin:$PATH' >> ~/.bashrc
     echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
     source ~/.bashrc
     ```
  4. Verified the installation:
     ```bash
     nvcc --version
     ```

---

## 4. Python Environment
- **Python Version**: 3.8
- **Steps Taken**:
  1. Created and activated a new Conda environment:
     ```bash
     conda create --name jukebox_env python=3.8 -y
     conda activate jukebox_env
     ```
  2. Installed PyTorch with CUDA 11.1:
     ```bash
     pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
     ```

---

## 5. Jukebox Installation
- **Steps Taken**:
  1. Set up the local repository:
     - Avoided recloning since the repository was already configured.
     - Verified the directory structure and existing files.
  2. Installed `jukebox` as an editable module:
     ```bash
     pip install -e .
     ```

---

## 6. Issues and Resolutions
1. **CUDA Compatibility**:
   - Installed CUDA 11.1 for compatibility with PyTorch 1.8.0.
   - Skipped unnecessary driver installation to avoid conflicts with the already installed NVIDIA driver.

2. **NumPy and Numba Conflict**:
   - Error: `AttributeError: module 'numpy' has no attribute 'long'`.
   - Resolution: Downgraded NumPy to version `1.19.5` for compatibility with Numba:
     ```bash
     pip install numpy==1.19.5
     ```

3. **Module Import Error**:
   - Error: `ModuleNotFoundError: No module named 'jukebox'`.
   - Resolution: Installed `jukebox` as an editable package with `pip install -e .`.

---

## 7. Verifications
1. Confirmed GPU and CUDA availability in PyTorch:
   ```bash
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
