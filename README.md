# ICP-Flow: LiDAR Scene Flow Estimation with ICP

Official implementation: [ICP-Flow: LiDAR Scene Flow Estimation with ICP](https://arxiv.org/abs/2402.17351) (CVPR 2024) 

[Yancong Lin](https://yanconglin.github.io/) and [Holger Caesar](https://sites.google.com/it-caesar.de/homepage/).

Intelligent Vehicles Group, Delft University of Technology, The Netherlands

Let op: code release in process, not ready to use yet.

## Introduction
Scene flow characterizes the 3D motion between two LiDAR scans captured by an autonomous vehicle at nearby timesteps. Prevalent methods consider scene flow as point-wise unconstrained flow vectors that can be learned by either large-scale training beforehand or time-consuming optimization at inference. However, these methods do not take into account that objects in autonomous driving often move rigidly. We incorporate this rigid-motion assumption into our design, where the goal is to associate objects over scans and then estimate the locally rigid transformations. We propose ICP-Flow, a learning-free flow estimator. The core of our design is the conventional Iterative Closest Point (ICP) algorithm, which aligns the objects over time and outputs the corresponding rigid transformations. Crucially, to aid ICP, we propose a histogram-based initialization that discovers the most likely translation, thus providing a good starting point for ICP. The complete scene flow is then recovered from the rigid transformations. We outperform state-of-the-art baselines, including supervised models, on the Waymo dataset and perform competitively on Argoverse-v2 and nuScenes. Further, we train a feedforward neural network, supervised by the pseudo labels from our model, and achieve top performance among all models capable of real-time inference. We validate the advantage of our model on scene flow estimation with longer temporal gaps, up to 0.4 seconds where other models fail to deliver meaningful results.

## Main Features: Motion Rigidity + Iterative Closesst Point

## Reproducing Results

### Installation

1. For easy reproducibility, you are suggested to install [miniconda](https://docs.conda.io/en/latest/miniconda.html) (or [anaconda](https://www.anaconda.com/distribution/) if you prefer) before following executing the following commands. 

```bash
conda create -f environment.yml
```
2. Install [Patchwork++](https://github.com/url-kaist/patchwork-plusplus) for ground segmentation, by following [patchwork-plusplus/README.md](patchwork-plusplus/README.md). It may take a while. There are several modifications on top of the original Patchwork++ to accommodate point indices.

### Processing the Dataset

Follow [Dynamic 3D Scene Analysis by Point Cloud Accumulation](https://github.com/prs-eth/PCAccumulation) to prepare the dataset, or you can simply run

```bash
wget --no-check-certificate --show-progress https://share.phys.ethz.ch/~gsg/PCAccumulation/data.zip
unzip data.zip
```

### Test
Configure all variables beforehand within the *.sh files.

```bash
bash main_waymo.sh
bash main_nuscenes.sh
```

Regarding the Argoverse-v2 scene flow dataset, see [ZeroFlow: Scalable Scene Flow via Distillation](https://github.com/kylevedder/zeroflow) for details, what I did was:

1. Run the ZeroFlow on the val/test set, and then save the points/labels in a .npz file one by one. See for details.
   
2. Configure all variables beforehand within the *.sh file.
```bash
bash main_argo.sh
```

### full results on Waymo,nuScenes and Argoverse-v2
Check *.txt.

You can download our reference pre-trained models (for the ICP-Flow + FNN model) from [SURFdrive](). 

### Pre-trained Models (optional, only for ICP-Flow + FNN)

You can download our reference pre-trained models (for the ICP-Flow + FNN model) from [SURFdrive](). 

### Demo

```Bash
bash demo.sh
```

### Citation
If you find our paper useful in your research, please consider citing:
```bash
@article{lin2024icp,
  title={ICP-Flow: LiDAR Scene Flow Estimation with ICP},
  author={Lin, Yancong and Caesar, Holger},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```
