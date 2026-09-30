# SCALE

SCALE improves lateral resolution uniformity in line-scanning confocal microscopy using training pairs derived from off-axis multi-line measurements. OS-SCALE further incorporates optical sectioning through dual-output prediction and differential processing.

### Software

* Python
* PyTorch
* NumPy
* tifffile
* MATLAB

### Hardware

* An NVIDIA GPU with CUDA support for network training and inference.
* Memory requirements depend on the dataset and processing settings.

## Instructions

Install PyTorch and the required Python dependencies. MATLAB is used for post-processing and deconvolution fusion.

### Usage

1. **Data preparation:** Configure `predata.py` and run `predata_main.py` in `py_code` to prepare training data from raw multi-line images. The input and target datasets are stored in `.npy` format.

2. **Training:** Configure the dataset and normalization settings in `data_loader.py`, then run `train.py`. Use the corresponding single-output or dual-output configuration for SCALE or OS-SCALE.

3. **Prediction:** Run `predict.py` with the trained model to process the test images. Network configuration and normalization should be consistent with training.

4. **Fusion:** Run `batch_fusion.m` in `matlab_code/deconv_fusion` for bidirectional deconvolution fusion using the corresponding directional PSFs. For OS-SCALE, first process the predicted dual-output images with `get_limo_from_limocestack.m`.

Data paths, acquisition settings, and processing parameters should be adapted to the dataset and imaging system.
