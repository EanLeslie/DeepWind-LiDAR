# DeepWind-LiDAR

Inference and example inputs for **DeepWind-LiDAR: Terrain-Aware Three-Dimensional Wind Field Reconstruction From a Single Scanning Doppler Lidar**.

DeepWind-LiDAR reconstructs the three wind components and turbulent kinetic energy on a regular three-dimensional grid.

## Setup

Use Python 3.11 and install the CPU inference dependencies:

```bash
python -m pip install -r requirements.txt
```

## Download the model

Download `deepwind-lidar.onnx` from the **Assets** section of [GitHub Releases](https://github.com/EanLeslie/DeepWind-LiDAR/releases).

Create a `models/` directory and place the downloaded file at `models/deepwind-lidar.onnx`, or provide its path with `--model`. The ONNX model includes input normalization and output denormalization.

## Run inference

Predictions are not included. Run inference on the prepared inputs to generate your own results in `outputs/`.

**Synthetic example (Fig. 7, Case A)**

```bash
python infer.py --input data/synthetic/cases/terrain_low/lidar_column_input.npy --output outputs/case_a.npz
```

**WFIP2 example (Fig. 11)**

```bash
python infer.py --input data/wfip2/20160130_081514/lidar_column_input.npy --output outputs/wfip2_20160130_081514.npz
```

Each command writes `u`, `v`, `w`, and `k` to an NPZ file. Each array has shape `(27, 300, 300)` in `(height, y, x)` order. Velocities are in m/s, and turbulent kinetic energy is in m²/s². Horizontal wind speed is `np.hypot(u, v)`. The height grid starts at 5 m AGL and runs in 5 m steps to 100 m, followed by 106.5, 114.95, 125.94, 140.22, 158.78, 182.91, and 214.29 m AGL.

## Height coordinates

`normalize/height.npy` contains the 27 output heights as a one-dimensional `float32` array. `normalize/height.csv` lists the same heights with zero-based level indices. Heights are in metres above local ground (AGL), following the FuXi-CFD released 27-level product. Element `i` corresponds to `u[i]`, `v[i]`, `w[i]`, and `k[i]`.

```python
import numpy as np

height_agl_m = np.load("normalize/height.npy", allow_pickle=False)
```

## Included examples

Each directory under `data/synthetic/cases/` contains a prepared input and its reference wind field:

- `lidar_column_input.npy`: Prepared model input.
- `reference.npz`: Reference wind field with `u`, `v`, `w`, and `k`.

Cases E and F contain identical copies of the same reference wind field.

| Fig. 7 example                      | Case directory under `data/synthetic/cases/` | Reference case ID |
| ----------------------------------- | ---------------------------------------------- | ----------------- |
| A: Lower terrain relief             | `terrain_low/`                               | `case_005926`   |
| B: Higher terrain relief            | `terrain_high/`                              | `case_005216`   |
| C: Lower reference wind speed       | `speed_low/`                                 | `case_009696`   |
| D: Higher reference wind speed      | `speed_high/`                                | `case_004471`   |
| E: Sparser observations             | `coverage_low/`                              | `case_005088`   |
| F: Original observations for Case E | `coverage_high/`                             | `case_005088`   |

The WFIP2 example (Fig. 11) is the scan from 2016-01-30 at 08:15:14 UTC. In `data/wfip2/20160130_081514/`:

- `lidar_column_input.npy` is the prepared model input.
- `observations/` contains the original WindTracer NetCDF scan and `observations.npz` with processed LOS observations.
- `paired_profiles.npz` contains the matched WindCube and Sodar measurements and the matching information.

## Input format

Provide a finite numeric NPY array with shape `(19, 300, 300)` in `(channel, y, x)` order. Channels are ordered as follows:

1. LOS velocity mean, standard deviation, minimum, and maximum.
2. Slant range mean, standard deviation, minimum, and maximum.
3. Height AGL mean, standard deviation, minimum, and maximum.
4. Mean beam-direction components in x, y, and z.
5. The log-transformed observation count.
6. The validity mask.
7. Terrain elevation and aerodynamic roughness length.

The bundled arrays are already encoded on the reconstruction grid. Pass them directly to the inference script.
