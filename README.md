# shrimpville

## Installation

1. Install clone sam2 from the repository.

```
git clone https://github.com/facebookresearch/sam2.git ; cd sam2
```
2. Create virtual environment for python as this will be easier to maintain between multiple people

```
python -m venv py-venv
```

3. Installing sam2 dependencies with the virtual python environment

```
./py-venv/bin/pip3 install -e . && ./py-venv/bin/pip3 install -e ".[notebooks]"
```
4. Install model checkpoints

```
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

5. Finish installing dependencies for other things

```
./py-venv/bin/pip3 install pandas
```

## Use
