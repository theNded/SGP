# Self-supervised Geometry Perception

## Disclaimer
In comparison to the code for the paper submission, this repository has been fully rewritten for a better readability and easier generalization. Please file a GitHub issue if there is anything buggy.

Since the final results are heavily depending on the RANSAC variations in geometry perception, we expect discrepancies comparing to the numbers in the paper. Again, please file an issue if a significant difference is observed.

### TODO
- [ ] 3D: Due to the changes of MinkowskiEngine and the Open3D RANSAC registration, current recall on 3DMatch is 2-3% lower than in the paper for all the methods (either supervised baseline or our self-supervised approach). Fix needed.
- [ ] 2D: Minimal test dataset and dataloader provided by the author of [CAPS](https://github.com/qianqianwang68/caps).
- [ ] Release Pretrained weights.

## Setup
Clone the project by 
```
git clone --recursive https://github.com/theNded/SGP.git
```
This will by default clone the submodules [FCGF](https://github.com/chrischoy/FCGF) and [CAPS](https://github.com/qianqianwang68/caps) for 3D and 2D perception, respectively. Please follow the instructions in the corresponding repositories to configure the submodule(s) of interest. 

## Datasets
For the 3D perception task, please download the [3DMatch dataset](https://drive.google.com/file/d/1P5xS4ZGrmuoElZbKeC6bM5UoWz8H9SL1/view) reorganized by us that aggregate point clouds by scenes. The reorganized [test set](https://drive.google.com/file/d/1AmmADbhk5X62Q6CnsbJcwm1BK0Uov1yG/view?usp=sharing) is also available.

For the 2D perception task, please download the [MegaDepth dataset](https://drive.google.com/file/d/1-o4TRLx6qm8ehQevV7nExmVJXfMxj657/view) provided by the author of CAPS.

## Vanilla training and testing
Copy and/or modify the `config_[train|test].yml` files in `perception3d`. The configurable parameters can be found in `perception3d/adaptor.py`. Then run
```
python perception3d/train.py --config /path/to/config.yml
python perception3d/test.py --config /path/to/config.yml
```
The same applies to 2D. 

Note our codebase is non-intrusive, i.e., the original repository are not modified, hence there are minor inconsistencies in configurations between 2D and 3D. Please carefuly check correspondent config options.


## Self-supervised training
The training runs in teacher-student meta loops, started with a bootstrap step (`bs`) supervised by hand-crafted features (SIFT/FPFH), followed by actual training loops (`00`, `01`) that trains a deep feature (CAPS/FCGF) by itself. After similarly configuring `config_sgp.yml`, run
```
python perception3d/sgp.py --config /path/to/config.yml
```
As the SGP process is time consuming, it is suggested to first perform a sanity check on a minimal set of data, configured in `config_sgp_sample.yml`.

To test the results per meta-iteration, by default run
```shell
# 2D
python perception2d/test.py --config perception2d/config_test.yml --ckpt_path caps_outputs/bs/caps_sgp/040000.pth
# 3D
python perception3d/test.py --config perception3d/config_test.yml --weights fcgf_outputs/bs/checkpoint.pth
```
for the trained feature from bootstrap (`bs`), and 
```shell
# 2D
python perception2d/test.py --config perception2d/config_test.yml --ckpt_path caps_outputs/00/caps_sgp/040000.pth
# 3D
python perception3d/test.py --config perception3d/config_test.yml --weights fcgf_outputs/00/checkpoint.pth
```
for the trained feature from 0-th meta-iteration (`00`) and following meta iterations.

To restart or extend current meta iterations, change `restart_meta_iter` and `max_meta_iters` in the configuration.

## Extension
To use your own dataset organized by scenes, checkout `dataset/`. A README details how the datasets are organized and how you may extend the base class and parse your scenes.

To train your own deep feature, checkout `sgp_base.py` and the corresponding `perception2d/` or `perception3d/` files. They share a similar interface for the `bootstrap` teaching-learning and `iterative` self-supervised teaching-learning.
