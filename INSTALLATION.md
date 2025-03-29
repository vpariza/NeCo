Installation instructions Posentangle
## Step 1: Create a python environment conda or just a python virtual environment
```bash
conda create -n "neco" python=3.11 ipython
```

## Step 2: Activate the conda environment
```bash
conda activate neco
```
or 
```bash
source activate neco
```

## Step 3: Update pip
```bash
python -m pip install --upgrade pip
```

## Step 4:  Install pytorch
Please  choose the appropriate cuda drivers for the specific version: https://pytorch.org/get-started/previous-versions/#v251
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```
## Step 5:  Install  Pytorch Lightning
We use pytorch lightning of `2.5.0.post`.
But you can use whatever is compatible with the specific torch version: https://lightning.ai/docs/pytorch/stable/versioning.html#compatibility-matrix
```bash
pip install lightning==2.5.0.post
```

## Step 6: Install Some other needed libraries
```bash
pip install click==8.1.8
pip install pandas==2.2.3
pip install sacred==0.8.7
pip install faiss-cpu==1.10.0
pip install timm==1.0.15
pip install scikit-learn==1.6.1
pip install scikit-image==0.25.2
pip install -U neptune==1.13.0
pip install diffsort==0.2.0
```

## Step 7: Install xformers to reduce execution time and memory usage when training or using dinov2 type of architectures
```bash
pip install xformers==0.0.29.post1
```