# shrimpville

## Installation

1. Clone sam2 from the repository.

```
git clone https://github.com/facebookresearch/sam2.git && cd sam2
```

2. Add path to sam2 to environment (i.e add to bashrc or zshrc)

```
echo "export SAM2_PATH=\"//home/noah/sam2/\"" >> ~/.zshrc && source ~/.zshrc
```

or for bash

```
echo "export SAM2_PATH=\"//home/$USER/sam2/\"" >> ~/.bashrc && source ~/.bashrc
```

3. Create virtual environment for python

```
python -m venv py-venv
```

4. Activate virt environment

```
source py-venv/bin/activate
```

5. Installing sam2 dependencies with the virtual python environment

```
pip3 install -e . && \
pip3 install -e ".[notebooks]"
```

6. Install model checkpoints

```
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

7. Finish installing dependencies used outside of sam2 requirements

```
pip3 install pandas
```

To deactivate the python virtual environment, simply run:
```
deactivate
```

## Usage
1. To run any of the python scripts, activate the python virtual environment:

```
source py-venv/bin/activate
```

2. Then if you need your video split into individaul frames run:

```
./extractFramesFromVid.sh videos/<file of video>
```

3. After this to get coordinates to start with run:

```
./extractCoord.py videos/<dir of jpgs created from previous script>
```

4. Then finaly run the main script to get csv

```
./extractMasks.py videos/<dir of jpgs created from previous script>
```

If you want to get the csv points ploted onto a grid to see the range of movement for the object simply run

```
./trackmotion.py videos/<dir of jpgs created from previous script>
```

To deactivate the python virtual environment, simply run:

```
deactivate
```
