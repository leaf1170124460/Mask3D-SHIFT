# Mask3D on SHIFT Dataset

This repository provides a multi task benchmark for instance segmentation, depth estimation, and 3D object detection. This repository also serves as a baseline method for the Multitask Robustness Challenge at the [VCL Workshop](https://wvcl.vis.xyz/challenges).

## Environment setup

Please make sure that conda or miniconda is installed on your machine before running the following command:

- Create a conda environment and activate it:
    ```
    conda create -n mmlab python=3.8
    conda activate mmlab
    conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
    ```

- Install mmcv-full and other packages:
    ```
    pip install -U openmim
    mim install mmcv-full==1.5.0 mmdet==2.25.0 mmdet3d==1.0.0rc3 mmsegmentation==0.27.0
    ```

- Install other dependencies:
    ```
    conda install protobuf
    pip install h5py==3.1.0
    pip uninstall setuptools -y
    pip install setuptools==59.5.0
    pip install tqdm
    pip install scalabel==0.3.0
    ```

Note: If there are other missing packages, please add them according to the error message.

## Download the SHIFT dataset

You can download the SHIFT dataset using the download script in [shift-dev](https://github.com/SysCV/shift-dev). Please follow the instructions below:

```shell
mkdir -p ./data/shift

# Download the discrete shift set for training source models
python download.py \
    --view "[front]" --group "[img, det_insseg_2d, depth, det_3d]" \
    --split "[train, val, minival]" \
    --framerate "[images]" \
    --shift "discrete" \
    ./data/shift
```
Here, we download the `minival` set, which is used exclusively for the Challenges at the [VCL Workshop](https://wvcl.vis.xyz/challenges).

## Training

- Download the pretrained weight [detr3d_resnet101.pth](https://drive.google.com/file/d/1YWX-jIS6fxG5_JKUBNVcZtsPtShdjE4O/view?usp=sharing).

- Customize input and output path:
    ```python
    # projects/configs/detr3d_r101_with_seg_depth_shift_minival.py line 7
    work_dir = '<YOUR_WORK_DIR>'
    
    # projects/configs/detr3d_r101_with_seg_depth_shift_minival.py line 144
    data_root = '<YOUR_DATA_ROOT>/shift/discrete/images/'
    
    # projects/configs/detr3d_r101_with_seg_depth_shift_minival.py line 258
    load_from = '<YOUR_CHECKPOINT_DIR>/detr3d_resnet101.pth'
    ```

- Run train script:
    ```shell
    bash tools/dist_train.sh projects/configs/detr3d_r101_with_seg_depth_shift_minival.py <NUM_GPU> --auto-resume
    ```

Note: We trained the baseline model on 2 NVIDIA RTX 3090, but 7GB of vRAM is sufficient for training.

## Testing

### Run from pretrained models

You are able to download the pretrained models and run the testing scripts directly.

- Download the pretrained model [epoch_10.pth](https://drive.google.com/file/d/1zHJrsYva8bb03pZTb_Vgsd8XcATfIzON/view?usp=sharing).
- Run test script:
    ```shell
    bash tools/dist_test.sh projects/configs/detr3d_r101_with_seg_depth_shift_minival.py <NUM_GPU> --checkpoint <CHECKPOINT_PATH> --out <OUT_PKL_PATH> --show-dir <OUT_RESULT_DIR> --format-only
    ```
- Zip the result folder if you want to submit a result.

## Evaluation

- Download the minival annotation [SHIFT_challenge2023_multitask.zip](https://github.com/suniique/SHIFT-3D-challenge/blob/challenge/annotations/SHIFT_challenge2023_multitask.zip).

- Run evaluation codes:

```python
python -m evaluation_script.main \
    --target <MINIVAL_ANNOTATION_PATH> \
    --pred <YOUR_RESULT_ZIP_PATH>
```

### Results on the minival set

<details>
<summary>
    <b>Instance Segmentation</b>
</summary>
<table>
    <tr>
        <td>mAP</td>
        <td>14.3516</td>
    </tr>
</table>
</details>

<details>
<summary>
    <b>Depth Estimation</b>
</summary>
<table>
    <tr>
        <td>SILog</td>
        <td>32.1495</td>
    </tr>
</table>
</details>

<details>
<summary>
    <b>3D object detection</b>
</summary>
<table>
    <tr>
        <td>mAP</td>
        <td>19.2338</td>
    </tr>
    <tr>
        <td>mTPS</td>
        <td>69.5190</td>
    </tr>
</table>
</details>

## License

Non-commercial. Code is heavily based on [Object DGCNN & DETR3D](https://github.com/WangYueFt/detr3d). 

