# FlyRobotv0.3 Hardware Test v0.1
After some small simulation testing, this soft will run on an "Arduino UNO Q, 4GB". the linux side runs the main.py, and the arduino side runs the sketch/sketck.ino. they communicate over uart using hardware/hardware.py
'''from arduino.router_bridge import Bridge   #Talks to the MCU sketch over Bridge RPC.'''



## Run
Install Python 3.10+ dependencies with `python -m pip install -r requirements.txt`.
Copy the six connectome data files listed below into `flybrain/`, then run `python main.py`.

Required `flybrain/` files:
- `ids.npy`, `indptr.npy`, `indices.npy`, `weights.npy`
- `body-annotations-male-cns-v1.0-minconf-0.5.feather`
- `body-neurotransmitters-male-cns-v1.0.feather`

