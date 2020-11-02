<h1 align="center">
CosyPose: Consistent multi-view multi-object 6D pose estimation
</h1>

<div align="center">
<h3>
<a href="http://ylabbe.github.io">Yann Labbé</a>,
<a href="https://jcarpent.github.io/">Justin Carpentier</a>,
<a href="http://imagine.enpc.fr/~aubrym/">Mathieu Aubry</a>,
<a href="http://www.di.ens.fr/~josef/">Josef Sivic</a>
<br>
<br>
ECCV: European Conference on Computer Vision, 2020
<br>
<br>
<a href="https://arxiv.org/abs/2008.08465">[Paper]</a>
<a href="https://www.di.ens.fr/willow/research/cosypose/">[Project page]</a>
<a href="https://youtu.be/4QYyEvnrC_o">[Video (1 min)]</a>
<a href="https://youtu.be/MNH_Ez7bcP0">[Video (10 min)]</a> 
<a href="https://docs.google.com/presentation/d/1APHpaKKnkIvmquNJUVqERiMN4gEQ10Jt4IY7wTfIVgE/edit?usp=sharing">[Slides]</a>
<br>
<br>
Winner of the <a href="https://bop.felk.cvut.cz/challenges/bop-challenge-2020/">BOP Challenge 2020 </a> at ECCV'20 <a href="https://docs.google.com/presentation/d/1jZDu4mw-uNcwzr5jMFlqEddZsb7SjQozXVG3dT6-1M0/edit?usp=sharing">[slides]</a>  <a href="https://arxiv.org/abs/2009.07378"> [BOP challenge paper] </a>
</h3>
</div>


# Citation
If you use this code in your research, please cite the paper:

```
@inproceedings{labbe2020,
title= {CosyPose: Consistent multi-view multi-object 6D pose estimation}
author={Y. {Labbe} and J. {Carpentier} and M. {Aubry} and J. {Sivic}},
booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
year={2020}}
```

# News
- CosyPose is the winning method in the [BOP challenge 2020](https://bop.felk.cvut.cz/challenges/) (5 awards in total, including best overall method and best RGB-only method) ! All the code and models used for the challenge are available in this repository.
- We participate in the [BOP challenge 2020](https://bop.felk.cvut.cz/challenges/bop-challenge-2020/). Results are available on the public [leaderboard](https://bop.felk.cvut.cz/leaderboards/) for 7 pose estimation benchmarks. We release 2D detection models (MaskRCNN) and 6D pose estimation models (coarse+refiner) used on each dataset.
- The paper is available on arXiv and full code is released.
- Our paper on CosyPose is accepted at ECCV 2020.

<!-- # TODO -->
<!-- - Add the script for visualization. -->
<!-- - Upload the BOP zip files to gdrive. -->

# Table of content
- [Overview](#overview)
- [Installation](#installation)
- [Downloading and preparing data](#downloading-and-preparing-data)
- [Note on GPU parallelization](#note-on-gpu-parallelization)
- [Reproducing single-view results](#reproducing-single-view-results)
- [Training the single-view 6D pose estimation models](#training-the-single-view-6D-pose-estimation-models)
  - [Synthetic data generation](#synthetic-data-generation)
  - [Training script](#training-script)
- [Reproducing multi-view results](#reproducing-multi-view-results)
- [Using CosyPose in a custom scenario](#using-cosypose-in-a-custom-scenario)
- [BOP20 models and results](#bop20-models-and-results)

# Overview
This repository contains the code for the full CosyPose approach, including:
### Single-view single-object 6D pose estimator
![Single view predictions](images/example_predictions.png)
  Given an RGB image and a 2D bounding box of an object with known 3D model, the 6D pose estimator predicts the full 6D pose of the object with respect to the camera. Our method is inspired from DeepIM with several simplications and technical improvements. It is fully implemented in pytorch and achieve single-view state-of-the-art on YCB-Video and T-LESS. We provide pre-trained models used in our experiments on both datasets. We make the training code that we used to train them available. It can be parallelized on multiple GPUs and multiple nodes.

###  Synthetic data generation
![Synthetic images](images/synthetic_images.png)
The single-view 6D pose estimation models are trained on a mix of synthetic and real images. We provide the code for generating the additionnal synthetic images.

### Multi-view multi-object scene reconstruction
![Multiview](images/multiview.png)

Single-view object-level reconstruction of a scene often fails because of detection mistakes, pose estimation errors and occlusions; which makes it inpractical for real applications. Our multi-view approach, CosyPose, addresseses these single-view limitations and helps improving 6D pose accuracy by leveraging information from multiple cameras with unknown positions. We provide the full code, including robust object-level multi-view matching and global scene refinement. The method is agnostic to the 6D pose estimator used, and can therefore be combined with many other existing single-view object pose estimation method to solve problems on other datasets, or in real scenarios. We provide a utility for running CosyPose given a set of input 6D object candidates in each image.


### BOP challenge 2020: single-view 2D detection + 6D pose estimation models
![BOP](images/bop_datasets.png)
We used our {coarse+refinement} single-view 6D pose estimation method in the [BOP challenge 2020](https://bop.felk.cvut.cz/challenges/bop-challenge-2020/). In addition, we trained a MaskRCNN detector (torchvision's implementation) on each of the 7 core datasets (LM-O, T-LESS, TUD-L, IC-BIN, ITODD, HB, YCB-V). We provide 2D detectors and 6D pose estimation models for these datasets. All training (including 2D detector), inference and evaluation code are available in this repository. It can be easily used for another dataset in the BOP format.


# Installation
```
git clone --recurse-submodules https://github.com/ylabbe/cosypose.git
cd cosypose
conda env create -n cosypose --file environment.yaml
conda activate cosypose
git lfs pull
python setup.py install
```
The installation may take some time as several packages must be downloaded and installed/compiled. If you plan to change the code, run `python setup.py develop`.

Notes:
- We use the [bop_toolkit](https://github.com/thodan/bop_toolkit) to compute some evaluation metrics on T-LESS. To ensure reproducibility, we use our [own fork](https://github.com/ylabbe/bop_toolkit_cosypose) of the repository. It is downloaded in `deps/`.


# Downloading and preparing data
<details>
<summary>Click for details...</summary>

All data used (datasets, models, results, ...) are stored in a directory `local_data` at the root of the repository. Create it with `mkdir local_data` or use a symlink if you want the data to be stored at a different place. We provide the utility `cosypose/scripts/download.py` for downloading required data and models. All of the files can also be [downloaded manually](https://drive.google.com/drive/folders/1JmOYbu1oqN81Dlj2lh6NCAMrC8pEdAtD?usp=sharing).

## BOP Datasets
For both T-LESS and YCB-Video, we use the datasets in the [BOP format](https://bop.felk.cvut.cz/datasets/). If you already have them on your disk, place them in `local_data/bop_datasets`. Alternatively, you can download it using :
```
python -m cosypose.scripts.download --bop_dataset=ycbv
python -m cosypose.scripts.download --bop_dataset=tless
```

Additionnal files that contain informations about the datasets used to fairly compare with prior works on both datasets.
```
python -m cosypose.scripts.download --bop_extra_files=ycbv
python -m cosypose.scripts.download --bop_extra_files=tless
```

We use [pybullet](https://pybullet.org/wordpress/) for rendering images which requires object models to be provided in the URDF format. We provide converted URDF files, they can be downloaded using:
```
python -m cosypose.scripts.download --urdf_models=ycbv
python -m cosypose.scripts.download --urdf_models=tless.cad
```

In the BOP format, the YCB objects `002_master_chef_can` and `040_large_marker` are considered symmetric, but not by previous works such as PoseCNN, PVNet and DeepIM. To ensure a fair comparison (using ADD instead of ADD-S for ADD-(S) for these objects), these objects must *not* be considered symmetric in the evaluation. To keep the uniformity of the models format, we generate a set of YCB objects `models_bop-compat_eval` that can be used to fairly compare our approach against previous works. You can download them directly:
```
python -m cosypose.scripts.download --ycbv_compat_models
```

Notes:
- The URDF files were obtained using these commands (requires `meshlab` to be installed):
```
python -m cosypose.scripts.convert_models_to_urdf --models=ycbv
python -m cosypose.scripts.convert_models_to_urdf --modles=tless.cad
```
- Compatibility models were obtained using the following script:
```
python -m cosypose.scripts.make_ycbv_compat_models
```

## Pre-trained models
The pre-trained models of the single-view pose estimator can be downloaded using:

```
# YCB-V Single-view refiner
python -m cosypose.scripts.download --model=ycbv-refiner-finetune--251020

# YCB-V Single-view refiner trained on synthetic data only 
# Only download this if you are interested in retraining the above model 
python -m cosypose.scripts.download --model=ycbv-refiner-syntonly--596719

# T-LESS coarse and refiner models 
python -m cosypose.scripts.download --model=tless-coarse--10219
python -m cosypose.scripts.download --model=tless-refiner--585928
```

## 2D detections
To ensure a fair comparison with prior works on both datasets, we use the same detections as DeepIM (from PoseCNN) on YCB-Video and the same as Pix2pose (from a RetinaNet model) on T-LESS. Download the saved 2D detections for both datasets using
```
python -m cosypose.scripts.download --detections=ycbv_posecnn

# SiSo detections: 1 detection with highest per score per class per image on all images
# Available for each image of the T-LESS dataset (primesense sensor)
# These are the same detections as used in Pix2pose's experiments
python -m cosypose.scripts.download --detections=tless_pix2pose_retinanet_siso_top1

# ViVo detections: All detections for a subset of 1000 images of T-LESS.
# Used in our multi-view experiments.
python -m cosypose.scripts.download --detections=tless_pix2pose_retinanet_vivo_all
```

If you are interested in re-training a detector, please see the BOP 2020 section.


Notes:
- The PoseCNN detections (and coarse pose estimates) on YCB-Video were extracted and converted from [these PoseCNN results](https://github.com/yuxng/YCB_Video_toolbox/blob/master/results_PoseCNN_RSS2018.zip).
- The Pix2pose detections were extracted using [pix2pose's](https://github.com/kirumang/Pix2Pose) code. We used the detection model from their paper, see [here](https://github.com/kirumang/Pix2Pose#download-pre-trained-weights). For the ViVo detections, their code was slightly modified. The code used to extract detections can be found [here](https://github.com/ylabbe/pix2pose_cosypose).

</details>

# Note on GPU parallelization
<details>
<summary>Click for details...</summary>

Training and evaluation code can be parallelized across multiple gpus and multiple machines using vanilla `torch.distributed`. This is done by simply starting multiple processes with the same arguments and assigning each process to a specific GPU via `CUDA_VISIBLE_DEVICES`. To run the processes on a local machine or on a SLUMR cluster, we use our own utility [job-runner](https://github.com/ylabbe/job-runner) but other similar tools such as [dask-jobqueue](https://github.com/dask/dask-jobqueue) or [submitit](https://github.com/facebookincubator/submitit) could be used. We provide instructions for single-node multi-gpu training, and for multi-gpu multi-node training on a SLURM cluster.

## Single gpu on a single node
```
# CUDA ID of GPU you want to use
export CUDA_VISIBLE_DEVICES=0
python -m cosypose.scripts.example_multigpu
```
where `scripts.example_multigpu` can be replaced by `scripts.run_pose_training` or `scripts.run_cosypose_eval` (see below for usage of training/evaluation scripts).

## Configuration of `job-runner` for multi-gpu usage
Change the path to the code directory, anaconda location and specify a temporary directory for storing job logs by modifying `job-runner-config.yaml'. If you have access to a SLURM cluster, specify the name of the queue, it's specifications (number of GPUs/CPUs per node) and the flags you typically use in a slurm script. Once you are done, run:
```
runjob-config job-runner-config.yaml
```

## Multi-gpu on a single node
```
# CUDA IDS of GPUs you want to use
export CUDA_VISIBLE_DEVICES=0,1
runjob --ngpus=2 --queue=local python -m cosypose.scripts.example_multigpu
```
The logs of the first process will be printed. You can check the logs of the other processes in the job directory.

## On a SLURM cluster
```
runjob --ngpus=8 --queue=gpu_p1  python -m cosypose.scripts.example_multigpu
```
</details>

# Reproducing single-view results
<details>
<summary>Click for details...</summary>

## YCB-Video
```
python -m cosypose.scripts.run_cosypose_eval --config ycbv
```
This will run the inference and evaluation on YCB-Video. We use our own implementation of the evaluation. We have checked that it matches the results from the original [matlab implementation](https://github.com/yuxng/YCB_Video_toolbox) for the AUC of ADD-S and AUC of ADD(-S) metrics. For example, you can see that the PoseCNN results are similar to the ones reported in the PoseCNN/DeepIM paper:
```
PoseCNN/AUC of ADD(-S): 0.613
```

The YCB-Video results and metrics can be downloaded directly:
```
python -m cosypose.scripts.download --result_id=ycbv-n_views=1--5154971130
```

## T-LESS
```
python -m cosypose.scripts.run_cosypose_eval --config tless-siso
```
This will run inference on the entire T-LESS dataset and print some metrics but not e_vsd<0.3 which is not supported in our code.
The results can also be downloaded:
```
python -m cosypose.scripts.download --result_id=tless-siso-n_views=1--684390594
```

To measure e_vsd<0.3, we use the BOP Toolkit. You can run it using:
```
python -m cosypose.scripts.run_bop_eval --result_id=tless-siso-n_views=1--684390594 --method=pix2pose_detections/refiner/iteration=4
```
This will create a `local_data/bop_predictions_csv/cosyposeXXXX-eccv2020_tless-test-primesense.csv` file in the BOP format and run evaluation. Intermediate metrics and final scores are saved in `local_data/bop_eval_outputs/cosposyXXXX-eccV2020_tless-test-primesense/`, where `XXXXX` correponds to a random number generated by the script.

The T-LESS SiSo results can also be downloaded directly:
```
python -m cosypose.scripts.download --bop_result_id=cosypose847205-eccv2020_tless-test-primesense
```

You can check the results match those from the paper:
```
cat local_data/bop_eval_outputs/cosypose847205-eccv2020_tless-test-primesense/error\=vsd_ntop\=1_delta\=15.000_tau\=20.000/scores_th\=0.300_min-visib\=0.100.json

{
  "gt_count": 69545,
  "mean_obj_recall": 0.6378486071644157,
  "mean_scene_recall": 0.6444110450903551,
  ...
  "recall": 0.632720209307857,
  ...
  "targets_count": 50452,
  "tp_count": 31922
}
```
Following other works, we reported `mean_obj_recall` in the paper.

## Single-view visualization
You can visualize the single-view predictions using [this](notebooks/visualize_singleview_predictions.ipynb) notebook as example.
</details>

# Training the single-view 6D pose estimation models
<details>
<summary>Click for details...</summary>

## Downloading synthetic images
The pose estimation models are trained on a mix of real images provided with the T-LESS/YCB-Video datasets and a set of images that we generated. For each dataset, we generated 1 million synthetic images. You can download these **large** datasets:
```
# 106 GB
python -m cosypose.scripts.download --synt_dataset=tless-1M

# 113 GB
python -m cosypose.scripts.download --synt_dataset=ycbv-1M
```
We provide below the instructions to generate these dataset locally if you are interested in using our synthetic data generation code.

## Synthetic data generation

### Textures for domain randomization
The synthetic training images are generated with some domain randomization. It includes adding textures to the background (and objects and T-LESS). We use a set of textures extracted from ShapeNet objects. Download the texture dataset:
```
python -m cosypose.scripts.download --texture_dataset
```

### Recording a synthetic dataset
The synthetic images are generated using multiple proceses managed by [dask](https://docs.dask.org/en/latest/setup/single-distributed.html). The synthetic training images can be generated using the following commands for both datasets:
```
export CUDA_VISIBLE_DEVICES=0
python -m cosypose.scripts.run_dataset_recording --config tless --local
python -m cosypose.scripts.run_dataset_recording --config ycbv --local
```
Make sure that enough space is available on your disk. We generate 1 million images which is around 120GB for each dataset. Note that we use a high number of synthetic images, but it may be possible to use fewer images. Please see directly the script `scripts/run_dataset_recording.py` for additionnal parameters. It is also possible to use [dask-jobqueue](https://jobqueue.dask.org/en/latest/) to generate the images on a cluster but we do not provide a simple configuration script for this at the moment. If you are interested in generating data using multiple machines on a cluster, you will have to modify dask-jobqueue's `Cluster` definition [here](cosypose/recording/record_dataset.py).

### Visualizing images of the dataset
You can visualize the images of the generated dataset using [this](notebooks/inspect_dataset.py) notebook. You can check that the ground truth prvided by a dataset is correct using [this](notebooks/render_dataset.py) notebook.


## Background images for data augmentation
We apply data augmentation to the training images. Data augmentation includes pasting random images of the pascal VOC dataset on the background of the scenes. You can download Pascal VOC using the following commands:
```
cd local_data
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar
```
(If the website is down, which happens periodically, you can alternatively download these files from  [a mirror](https://pjreddie.com/projects/pascal-voc-dataset-mirror/) at https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar)

## Training script
Once you have generated the synthetic data images and downloaded pascal VOC, you can run the training script. On YCB-Video, we train a coarse model on synthetic data only and fine-tune it on the synthetic + real images. On T-LESS, we train a coarse and refinement model and synthetic + provided real images of isolated objects directly from scratch. In our experiments, all models are trained using the same procedure on 32 GPUs.

```
runjob --ngpus=32 python -m cosypose.scripts.run_pose_training --config ycbv-refiner-syntonly
runjob --ngpus=32 python -m cosypose.scripts.run_pose_training --config ycbv-refiner-finetune
runjob --ngpus=32 python -m cosypose.scripts.run_pose_training --config tless-coarse
runjob --ngpus=32 python -m cosypose.scripts.run_pose_training --config tless-refiner
```

You can visualize the logs of the provided models in [this](notebooks/paper_training_logs.ipynb) notebook. 

![Logs](images/screenshot_logs.png)

You can add the `run_id` of each model that are your are training to visualize training metrics.

Notes:
- While we used 32 GPUs in our experiments, the training script can be ran with any number of GPUs. It will just be slower and the overall batch size will be smaller. We have not studied the impact of batch size on final performance of the model. On 32 NVIDIA V100, training a model takes approximately 10 hours. Note that the models are trained from scratch on all the objects of each dataset simulatenously.
- If you are interested in training with limited resources, you could consider the following changes to the code: (a) use a smaller backbone, e.g. flownet, resnet18 or resnet34, (b) train for fewer iterations, (c) start from one of our pre-trained models. All the parameters are defined in `cosypose/scripts/run_pose_training.py`. If you are trying to train with limited resources or on your own dataset and datas, please do not hesitate to share your experience, by opening an issue or by sending an email !
- We run evaluation of the models a few times during training. You can disable it by adding the flag `--no-eval` to speed up training. Note that the we do not use the evaluation metrics to find the best model since no official validation splits are available for YCB-Video/T-LESS. We always report results for the model obtained at the end of the training.

</details>

# Reproducing multi-view results
<details>
<summary>Click for details...</summary>

The following scripts will run the full CosyPose pipeline (single-view predictions + multi-view scene reconstruction), compute the metrics reported in the paper and save the results to a directory in `local_data/results/`.
```
export CUDA_VISIBLE_DEVICES=0
python -m cosypose.scripts.run_cosypose_eval --config tless-vivo --nviews=4
python -m cosypose.scripts.run_cosypose_eval --config tless-vivo --nviews=8
python -m cosypose.scripts.run_cosypose_eval --config ycbv --nviews=5
```
Note that the inference and evaluation can be sped up using `runjob` if you have access to multiple GPUs. The mAP@ADD-S<0.1d and AUC of ADD-S metrics are computed using our own code since they are not supported by the BOP toolkit. We refer to the appendix of the main paper for more details on these metrics.

The results can be also downloaded directly:
```
# YCB-Video 5 views
python -m cosypose.scripts.download --result_id=ycbv-n_views=5--8073381555 

# T-LESS ViVo 4 views
python -m cosypose.scripts.download --result_id=tless-vivo-n_views=4--2731943061

# T-LESS ViVo 8 views
python -m cosypose.scripts.download --result_id=tless-vivo-n_views=8--2322743008
```

On T-LESS ViVo, the evsd<0.3 and ADD-S<0.1d metrics are computed using the BOP toolkit, for example for computing the multi-view results for ViVo 8 views:
```
python -m cosypose.scripts.run_bop_eval  --results  tless-vivo-n_views=8--2322743008 --method pix2pose_detections/ba_output+all_cand --vivo
```
The `ba_output+all_cand` predictions correspond to the output of CosyPose concatenated to all the single-view candidates as explained in the experiment section of the paper. The single-view candidates have strictly lower score than the multi-view predictions, which means that single-view estimates are used for evaluation only if there are no multi-view predictions for an object, e.g. typically because a camera cannot be placed with respect to the scene because there are too few inlier candidates.

We also provide the BOP evaluation results that we computed and reported in the paper:
```
# T-LESS ViVo 1 view
python -m cosypose.scripts.download --bop_results=cosypose68486-eccv2020_tless-test-primesense

# T-LESS ViVo 4 views
python -m cosypose.scripts.download --bop_results=cosypose615294-eccv2020_tless-test-primesense

# T-LESS ViVo 8 views
python -m cosypose.scripts.download --bop_result_id=cosypose114533-eccv2020_tless-test-primesense
```

## Multi-view visualization
You can use [this](notebooks/visualize_multiview_predictions.ipynb) notebook to visualize the multi-view results on YCB-Video and T-LESS and generate the 3D visualization GIFs.

![plots_cosypose](images/screenshot_plots_cosypose.png)

![GIF](notebooks/gifs/scene_ds=tless.primesense.test.bop19-scene=16-nviews=8-scene_group=105.gif)


</details>

# Running CosyPose in a custom scenario
<details>
<summary>Click for details...</summary>

Stage 2 and 3 of CosyPose are agnostic to the 6D pose estimator used, and can therefore be combined with many other existing single-view object pose estimation method to solve problems on other datasets, or for real applications. We provide a utility for running CosyPose given a set of input 6D object candidates in each image.

If you are willing to combine CosyPose with your own pose estimator, you will need to provide the following:
- The 3D models of the objects considered and their associated symmetries. The models should be provided in a format similar to the BOP format in a `models` directory.
- A set of input 6D object candidates in each image `candidates.csv`. We use the same convention as the BOP format, but all the candidates in this file must be provided for a unique scene (a single 3D reconstruction) in different views.
- The intrinsics parameters of the cameras of each view in a file `scene_camera.json` following the BOP format.
<!-- - [Optional], a `urdfs` directory which contains the `models` converted to `urdfs` (using only objs mesh for pybullet). for visualization -->

Use these commands to create a custom scenario with T-LESS objects and run CosyPose on it:
```
cd local_data
mkdir -p custom_scenarios/example
ln -s $(pwd)/bop_datasets/tless/models custom_scenarios/example

export CUDA_VISIBLE_DEVICES=0
python -m cosypose.scripts.download --example_scenario
python -m cosypose.scripts.run_custom_scenario --scenario=example
```

This will generate the following files:
- `results/subscene=0/predicted_scene.json` a set of predicted objects and cameras with their associated poses in a common reference frame.
- `results/subscene=0/scene_reprojected.csv` poses of predicted objects expressed in camera frames, in the BOP format.
<!-- - [If urdfs are provided] `results/subscene=0/visualizations.png` a figure which shows the input candidates and the projected scene in each view. -->
<!-- - [If urdfs are provided] `results/subscene=0/visualization.gif` a visualization GIF of the predicted scene in 3D. -->

You can use this as an example to check the different formats in which the informations should be provided.

Notes:
- This is experimental. The default parameters for the pipeline should give good results in many scenarios (we use the same on YCB-Video and T-LESS) but we have yet not conducted experiments in many custom scenarios. If you are trying to apply CosyPose to your own 6D pose estimations and encounter any issues or would like to obtain better results, please consider sharing your experience, I would be very happy to help you.

- The script is quite slow to run for a single scene because all models need to be loaded and the first cuda call with pytorch is always slow. If you would like to use this for an application, consider using directly the API of the `MultiviewScenePredictor` in your own code. You can use the script `scripts/run_custom_scenario.py` as an example on how to use it.

- While the object candidate matching stage (stage 2 of CosyPose) is fairly optimized using a combination of C++ and fully batched operations in pytorch on GPU, the stage 3 (global scene refinement via object-level bundle adjustment) could largely be sped up. The implementation would benefit from being written in C++ with a standard optimization framework instead of using pytorch to compute full jacobians for Levenberg-Marquart. If you find this stage to be too slow for your problem/application, please open an issue or let me know. If there is demand for a faster implementation, I may provide it in the future.

</details>

# BOP20 models and results
<details>
<summary>Click for details...</summary>

We provide the training code that we used to train single-view single-object pose estimation models on the 7 core datasets (LM-O, TLESS, TUD-L, IC-BIN, ITODD, HB, YCB-V) and pre-trained detector and pose estimation models. Note that these models are different from the ones used in the paper. The differences with the models used in the paper are the following:
- In the paper, we use already available detectors for T-LESS and YCB-Video. For the BOP20 challenge, we trained our own detectors on each dataset.
- Detection and pose estimation models are trained using PBR synthetic images provided with the BOP challenge instead of using our own synthetic data to make it easier to compare fairly with the other approaches.
- In the BOP20 challenge results, the initialization of the pose provided to the coarse model is slightly different. First, the canonical orientation has been changed to have the z-axis parallel to the camera instead of having the x-axis parallel to the camera, a position with z-axis upward and parallel to the camera makes the overall shape and details of the objects more visible. Second, instead of fixing the z value of the canonical translation to 1 meter, we compute a guess of object depth using the height and width of the 2D bounding box and the 3D model. This makes the method more general as the canonical depth is always within a reasonable range of the correct depth even if the object is very far from the camera.

Even though the challenge is focused on single-view pose estimation, we also reported multi-view results on YCB-Video, T-LESS and HB for 4 and 8 views.

## Downloading BOP datasets
```
python -m cosypose.scripts.download --bop_dataset=DATASET --pbr_training_images
python -m cosypose.scripts.download --urdf_models=DATASET
```
for DATASET={hb,icbin,itodd,lm,lmo,tless,tudl,ycbv}. If you are not interested in training the models, you can remove the flag --pbr_training_images and you can omit lm.

## Pre-trained models
You can download all the models that we trained for the challenge using our downloading script:
```
python -m cosypose.scripts.download --model=model_id
```
where model_id is given by the table below:


| Dataset | Model type | Training images | `model_id`                           |
|---------|------------|-----------------|--------------------------------------|
| hb      | detector   | PBR             | detector-bop-hb-pbr--497808          |
| hb      | coarse     | PBR             | coarse-bop-hb-pbr--7075              |
| hb      | refiner    | PBR             | refiner-bop-hb-pbr--247731           |
|         |            |                 |                                      |
| icbin   | detector   | PBR             | detector-bop-icbin-pbr--947409       |
| icbin   | coarse     | PBR             | coarse-bop-icbin-pbr--915044         |
| icbin   | refiner    | PBR             | refiner-bop-icbin-pbr--841882        |
|         |            |                 |                                      |
| lmo     | detector   | PBR             | detector-bop-lmo-pbr--517542         |
| lmo     | coarse     | PBR             | coarse-bop-lmo-pbr--707448           |
| lmo     | refiner    | PBR             | refiner-bop-lmo-pbr--325214          |
|         |            |                 |                                      |
| itodd   | detector   | PBR             | detector-bop-itodd-pbr--509908       |
| itodd   | coarse     | PBR             | coarse-bop-itodd-pbr--681884         |
| itodd   | refiner    | PBR             | refiner-bop-itodd-pbr--834427        |
|         |            |                 |                                      |
| tless   | detector   | PBR             | detector-bop-tless-pbr--873074       |
| tless   | coarse     | PBR             | coarse-bop-tless-pbr--506801         |
| tless   | refiner    | PBR             | refiner-bop-tless-pbr--233420        |
| tless   | detector   | SYNT+REAL       | detector-bop-tless-synt+real--452847 |
| tless   | coarse     | SYNT+REAL       | coarse-bop-tless-synt+real--160982   |
| tless   | refiner    | SYNT+REAL       | refiner-bop-tless-synt+real--881314  |
|         |            |                 |                                      |
| tudl    | detector   | PBR             | detector-bop-tudl-pbr--728047        |
| tudl    | coarse     | PBR             | coarse-bop-tudl-pbr--373484          |
| tudl    | refiner    | PBR             | refiner-bop-tudl-pbr--487212         |
| tudl    | detector   | SYNT+REAL       | detector-bop-tudl-synt+real--298779  |
| tudl    | coarse     | SYNT+REAL       | coarse-bop-tudl-synt+real--610074    |
| tudl    | refiner    | SYNT+REAL       | refiner-bop-tudl-synt+real--423239   |
|         |            |                 |                                      |
| ycbv    | detector   | PBR             | detector-bop-ycbv-pbr--970850        |
| ycbv    | coarse     | PBR             | coarse-bop-ycbv-pbr--724183          |
| ycbv    | refiner    | PBR             | refiner-bop-ycbv-pbr--604090         |
| ycbv    | detector   | SYNT+REAL       | detector-bop-ycbv-synt+real--292971  |
| ycbv    | coarse     | SYNT+REAL       | coarse-bop-ycbv-synt+real--822463    |
| ycbv    | refiner    | SYNT+REAL       | refiner-bop-ycbv-synt+real--631598   |


The detectors are MaskRCNN models with resnet50 FPN backbone. PBR corresponds to training only on provided synthetic images. SYNT+REAL corresponds to training on all available synthetic and real images when available (only for tless, tudl and ycbv). SYNT+REAL models are pre-trained from PBR.

If you want to use all the models for a complete evaluation:
```
python -m cosypose.scripts.download --all_bop20_models
```

## Running inference
The following commands will reproduce the results that we reported on the [leaderboard](https://bop.felk.cvut.cz/leaderboards/) for all the datasets:
```
# CosyPose-ECCV20-PBR-1VIEW	
python -m cosypose.scripts.run_bop_inference --config bop-pbr

# CosyPose-ECCV20-SYNT+REAL-1VIEW
python -m cosypose.scripts.run_bop_inference --config bop-synt+real

# CosyPose-ECCV20-SYNT+REAL-1VIEW-ICP	
python -m cosypose.scripts.run_bop_inference --config bop-synt+real --icp

# CosyPose-ECCV20-SYNT+REAL-4VIEWS	
python -m cosypose.scripts.run_bop_inference --config bop-synt+real --nviews=4

# CosyPose-ECCV20-SYNT+REAL-8VIEWS	
python -m cosypose.scripts.run_bop_inference --config bop-synt+real --nviews=8
```
The inference script is compatible with `runjob`.

Inference results on all datasets can be downloaded directly:
```
python -m cosypose.scripts.download --result_id=result_id
```
where result_id is given by the table below

| BOP20 method name                   | `result_id`                    |
|-------------------------------------|--------------------------------|
| CosyPose-ECCV20-PBR-1VIEW           | bop-pbr--223026                |
| CosyPose-ECCV20-SYNT+REAL-1VIEW     | bop-synt+real--815712          |
| CosyPose-ECCV20-SYNT+REAL-1VIEW-ICP | bop-synt+real-icp--121351      |
| CosyPose-ECCV20-SYNT+REAL-4VIEWS    | bop-synt+real-nviews=4--419066 |
| CosyPose-ECCV20-SYNT+REAL-8VIEWS    | bop-synt+real-nviews=8--763684 |

If you want to download everything:
```
python -m cosypose.scripts.download --all_bop20_results
```

Notes:
- The ICP refiner was adapted from [Pix2Pose code](https://github.com/kirumang/Pix2Pose/blob/843effe0097e9982f4b07dd90b04ede2b9ee9294/tools/5_evaluation_bop_icp3d.py#L57). Be careful if you want to use it, it slightly decrease performance over RGB-only on T-LESS instead of improving the results. Qualitative results show a misalignment of many objects after ICP, there is likely a small bug with my version but I haven't had time to go in detail. Note that our method and paper are focused on the RGB-only setting.

<!-- ## Visualizing results -->
<!-- TODO -->
<!-- Inference results can be visualized on each dataset, please see this notebook for an example. -->

## Running evaluation
You can run locally the evaluation on the publicly available test sets:
```
python -m cosypose.scripts.run_bop20_eval_multi --result_id=result_id --method=method
```
where method is `maskrcnn_detections/refiner/iteration=4` for single-view, `maskrcnn_detections/icp` when ICP is ran, and `maskrcnn_detections/multiview` for multi-view (n_views > 1).

If you are only interested in generating the bop predictions file suitable for submission to the website, you can run
```
python -m cosypose.scripts.run_bop20_eval_multi --result_id=result_id --method=method --convert_only
```

## Training details

### Detection
We use torchvision's MaskRCNN implementation for the detection. The models were trained using:
```
runjob --ngpus=32 python -m cosypose.scripts.run_detector_training --config bop-DATASET-TRAINING_IMAGES
```
where DATASET={lmo,tless,tudl,icbin,itodd,hb,ycbv} and TRAINING_IMAGES={pbr,synt+real} (synt+real only for datasets where real images are available: tless, tudl and ycbv).

### Pose estimation
```
runjob --ngpus=32 python -m cosypose.scripts.run_pose_training --config bop-DATASET-TRAINING_IMAGES-MODEL_TYPE
```
where MODEL_TYPE={coarse,refiner}.

</details>

<!-- TODO -->
<!-- Training logs are available in [this](notebooks/bop20_training_logs.ipynb) notebook. -->
