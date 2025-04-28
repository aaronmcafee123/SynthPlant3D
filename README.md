# SynthPlant3D




To install dependencies run:

You might have to jimmy this script to get it work.

```bash
wget https://github.com/IntelRealSense/librealsense/raw/master/scripts/libuvc_installation.sh
chmod +x ./libuvc_installation.sh
./libuvc_installation.sh -DBUILD_PYTHON_BINDINGS:bool=true
```

Then you'll want to use the command:

```bash
pip install -r requirements.txt
```

For RadFoam models of hte blueberry and oat plant, please use the google drive link below, this also contains the raw image data used to train these models
https://drive.google.com/drive/folders/1iYZ0QNN5ANqcA-h6fNQK_xVrXyBWERZZ
