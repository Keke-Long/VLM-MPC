# VLM-MPC

Keke Long, Haotian Shi, Jiaxi Liu, Chaowei Xiao, and Xiaopeng Li, "VLM-MPC: Model Predictive Controller Augmented Vision Language Model for Autonomous Driving," *Transportation Research Part C: Emerging Technologies*, vol. 183, 2026, Article 105487, DOI: [10.1016/j.trc.2025.105487](https://doi.org/10.1016/j.trc.2025.105487).

## Overview

VLM-MPC is a closed-loop autonomous-driving framework that combines a Vision-Language Model (VLM) with Model Predictive Control (MPC).

The framework uses two asynchronous levels:

1. **Upper-level VLM**: Interprets driving scenes and generates driving parameters, such as desired speed and desired headway, using front-camera images, ego-vehicle states, environmental conditions, and reference memory.
2. **Lower-level MPC**: Uses those parameters to control the vehicle in real time while accounting for vehicle dynamics and engine lag, and returns state feedback to the overall system.

The corresponding article evaluates VLM-MPC using the nuScenes dataset and CARLA simulation across challenging conditions including night, rain, fog, and intersections. The experiments examine safety, smoothness, environmental understanding, and the response stability provided by the reference-memory and environment-encoder components.

## Repository Structure

The repository includes materials for:

- **VLM-based decision making**: VLM/LLM controllers and scene reasoning.
- **Model Predictive Control**: MPC calibration and lower-level vehicle control.
- **Data preparation**: Processing of scene, image, and vehicle-state data.
- **Image processing**: Preparation of visual inputs for the VLM.
- **Evaluation**: Safety, smoothness, and response-stability evaluation.
- **Baseline comparison**: Materials for comparison with Agent-Driver and related baselines.
- **Example scenes and figures**: Sample inputs and framework illustrations.

## Installation

Create a Python environment and install the listed dependencies:

```bash
git clone https://github.com/Keke-Long/VLM-MPC.git
cd VLM-MPC
pip install -r requirements.txt
```

Some components may require access credentials for external model APIs. Users should provide their own credentials through environment variables or local configuration and should not commit credentials to the repository.

## Citation

If you use this repository, please cite:

```bibtex
@article{long2026vlmmpc,
  title={VLM-MPC: Model predictive controller augmented vision language model for autonomous driving},
  author={Long, Keke and Shi, Haotian and Liu, Jiaxi and Xiao, Chaowei and Li, Xiaopeng},
  journal={Transportation Research Part C: Emerging Technologies},
  volume={183},
  pages={105487},
  year={2026},
  doi={10.1016/j.trc.2025.105487}
}
```

## License

This project is released under the [Apache License 2.0](LICENSE).
