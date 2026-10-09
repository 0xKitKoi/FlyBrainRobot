# FlyRobot Starter

Starter scaffold for the fly-brain robot. It separates the connectome model, virtual sensors, behavior selection, simulated servo poses, and future hardware integration.

## Run
Install Python 3.10+ dependencies with `python -m pip install -r requirements.txt`.
Copy the six connectome data files listed below into `flybrain/`, then run `python main.py`.

Required `flybrain/` files:
- `ids.npy`, `indptr.npy`, `indices.npy`, `weights.npy`
- `body-annotations-male-cns-v1.0-minconf-0.5.feather`
- `body-neurotransmitters-male-cns-v1.0.feather`

Commands: `l` left touch, `r` right touch, `b` both, `h` warm target, `m` arm motion, `n` neutral, `status`, `q` quit.

Important: the uploaded script is an experiment, not a real-time motor controller. This refactor keeps its simplified left/right haltere-drive experiment. Touch and heat are NOT falsely injected into haltere neurons; they currently affect the explicit behavior layer. Servo poses are simulated only. No hardware is controlled yet.
