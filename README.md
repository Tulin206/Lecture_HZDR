#### Please set up your Python environment and install all required packages before the course begins. During the course sessions, there will not be time allocated for local environment setup or package installation. Preparing your environment in advance ensures you can follow along and participate fully.

### Schedule for HPCGateway Fall School: https://notes.desy.de/qoMPic6oQ36VhxvivBVFCg?both#Time-Schedule
### Schedule for HIDA: https://notes.desy.de/YhM3gdx-QqKJO0hAR9SyvQ?both



# Deep Learning Project Setup and Usage Guide

This repo contains a Jupyter Book with notebooks runnable in local environment and also in Google Colab.


## Setup (local)

This project uses [uv](https://github.com/astral-sh/uv?tab=readme-ov-file#python-versions) for fast Python environment management and dependency installation.

Follow the installation instructions for `uv` at the link above or you can follow the below instructions.

Requires Python 3.9+ (recommended: a virtual environment).

```bash
# 1) Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install uv

# 2) Install the package (includes notebook/runtime deps like numpy, torch, ipykernel)
uv pip install -e .

# 3) Optional: developer tools (linting, testing)
uv pip install -e .[dev]

# 4) Optional: build the Jupyter Book site
uv pip install -e .[book]
jupyter-book build book/
```

Notes

* Torch CPU wheels are installed by default. GPU acceleration may require a different wheel/index per your platform—see PyTorch.org for instructions.
* After installing the dev extra, you can run tests with `pytest`.

After finishing the local setup, you can run the code using the provided `.py` files directly. Alternatively, you can use the `.ipynb` Jupyter notebooks if you install Jupyter in your IDE (e.g., with `pip install notebook` or `pip install jupyterlab`). In both cases, make sure to install all dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Google Colab Usage (Recommended for Limited Local Hardware)
> **Note:** If you are comfortable using your local machine and are familiar with Python environment managers such as `uv`, `conda`, or `virtualenv`, you can skip the Google Colab instructions below and follow the local setup instructions provided earlier in this README.

**If your local machine does not have sufficient hardware (CPU/GPU/TPU) for deep learning, we recommend using Google Colab. Colab provides free access to powerful hardware accelerators and requires no local setup.**

If you want to manually use Google Colab, you can follow the instructions below. Otherwise, simply use the Open in Colab badges at the top of each notebook for the easiest experience.

### Initial Setup in Google Colab

1. **Mount Google Drive:**
   - After uploading your notebook, you need to mount your Google Drive
   - Run the following code in a cell:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```
   - Click the authorization link and follow the prompts
   - After authorization, your Google Drive will be mounted and accessible

2. **Set Up Python Path:**
   - First, copy the notebook to your Google Drive's "Colab Notebooks" folder
   - Then run the following code to set up the correct paths:
     ```python
     import os, sys
     BASE = '/content/drive/MyDrive/Colab Notebooks'  # this can have spaces
     os.chdir(BASE)             # go into the folder
     sys.path.insert(0, BASE)   # add it to Python path
     ```

3. **Verify Setup:**
   - In the left sidebar, you should see "MyDrive"
   - Under MyDrive, look for the "Colab Notebooks" folder
   - Make sure your notebook is visible in this folder

### Using Notebooks in Google Colab

1. **Access Google Colab:**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Sign in with your Google account

2. **Upload and Open Notebooks:**
   - Click on `File` > `Upload notebook`
   - Select the `.ipynb` file you want to work with
   - Or use `File` > `Open notebook` > `Upload` to browse your local files

3. **Working with the Notebooks:**
   - Make sure to run the setup cells at the beginning of each notebook
   - For notebooks requiring external data files, you'll need to upload them to Colab's runtime
   - Use the `Runtime` menu to:
     - Run all cells
     - Restart runtime
     - Change runtime type (if you need GPU acceleration)

4. **Save Your Work:**
   - Save a copy to your Google Drive: `File` > `Save a copy in Drive`
   - Download locally: `File` > `Download` > `Download .ipynb`

## Google Colab Hardware Acceleration

Google Colab provides different hardware accelerators to speed up model training:

### Types of Hardware Accelerators

1. **CPU (Central Processing Unit)**
   - Default option
   - Suitable for basic computations and small models
   - Slowest among the three options
   - Use when:
     - Running basic data preprocessing
     - Training very small models
     - Testing code functionality

2. **GPU (Graphics Processing Unit)**
   - Significantly faster than CPU for deep learning
   - Ideal for most deep learning tasks
   - NVIDIA GPUs (Tesla T4, P100, or V100)
   - Use when:
     - Training neural networks
     - Processing image/video data
     - Running parallel computations
   - Free tier limitations apply

3. **TPU (Tensor Processing Unit)**
   - Google's custom-designed AI accelerator
   - Fastest option for specific models
   - Best for TensorFlow models
   - Use when:
     - Training large TensorFlow models
     - Need maximum performance
     - Working with distributed training
   - May require code modifications

### How to Enable Hardware Acceleration in Colab

1. Click on "Runtime" in the top menu
2. Select "Change runtime type"
3. Choose your hardware accelerator:
   - None (CPU)
   - GPU
   - TPU
4. Click "Save"

### Verifying GPU/TPU Connection

For GPU:
```python
import tensorflow as tf
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

# Or for PyTorch
import torch
print("GPU Available: ", torch.cuda.is_available())
print("GPU Device Name: ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

For TPU:
```python
import tensorflow as tf
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
except ValueError:
    print("No TPU detected")
```

### Best Practices

1. **Resource Management**:
   - Free tier has usage limits
   - Sessions disconnect after 12 hours
   - Save your work frequently
   - Use `runtime.reset()` to clear memory

2. **Choosing Accelerator**:
   - CPU: Data preprocessing, small models
   - GPU: Most deep learning tasks, PyTorch models
   - TPU: Large TensorFlow models, distributed training

3. **Memory Usage**:
   - Monitor memory usage (RAM)
   - Clear output cells when not needed
   - Restart runtime if memory issues occur
   - Use appropriate batch sizes

4. **Performance Tips**:
   - Keep data on Google Drive for faster access
   - Use efficient data loading methods
   - Enable mixed precision training when possible
   - Monitor training with tensorboard

## Note
- The notebooks are self-contained and include all necessary package installations
- Some notebooks may require additional data files - check the notebook contents for specific requirements
- Make sure to select GPU runtime in Colab for notebooks involving deep learning models

---
