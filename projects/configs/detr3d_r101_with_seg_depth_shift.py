_base_ = [
    '../../mmdetection3d/configs/_base_/default_runtime.py'
]

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
work_dir = '<YOUR_WORK_DIR>'

# If point cloud range is changed, the models should also change their point cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
depth_bound = (1, 200, 1)

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

class_names = [
    "pedestrian", "car", "truck", "bus", "motorcycle", "bicycle"
]

data_config = {
    'cams': ['CAM_FRONT'],
    'Ncams': 1,
    'input_size': (800, 1280),
    'src_size': (800, 1280),

    # No augmentation
    'resize': (0, 0),
    'rot': (0, 0),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

model = dict(
    type='Detr3D_Seg_Depth',
    use_grid_mask=True,
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    img_feats_levels=(0, 1, 2, 3),
    pts_bbox_head=dict(
        type='Detr3DHead_Seg_Depth',
        num_cams=1,
        num_query=200,
        with_stuff=False,
        num_classes=6,
        in_channels=256,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='Detr3DTransformer',
            decoder=dict(
                type='Detr3DTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='Detr3DCrossAtten',
                            pc_range=point_cloud_range,
                            num_points=1,
                            embed_dims=256)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=200,
            voxel_size=voxel_size,
            num_classes=6),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_depth=dict(
            type='DepthLoss',
            loss_weight=3.0,
            depth_act_mode='sigmoid',
            si_weight=1.0,
            sq_rel_weight=1.0,
            abs_rel_weight=1.0,
            dbound=depth_bound, ),
        dbound=depth_bound),

    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='Unified_HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),  # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range),
    )))

dataset_type = 'SHIFTDataset'
data_root = '<YOUR_DATA_ROOT>/shift/discrete/images/'

file_client_args = dict(backend='disk')

meta_keys = ('ori_shape', 'img_shape', 'lidar2img',
             'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
             'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
             'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
             'pcd_scale_factor', 'pcd_rotation', 'pcd_rotation_angle',
             'pts_filename', 'transformation_3d_flow', 'trans_mat',
             'affine_aug', 'video_name', 'image_name')

train_pipeline = [
    dict(type='LoadAnnotations3D',
         with_bbox=False,
         with_mask=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='Load_Annotations', with_bbox_3d=True, with_label_3d=True, with_attr_label=False, with_mask_2d=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='FormatBundle3D', class_names=class_names, with_mask=True, dataset_type=dataset_type),
    dict(type='GenerateAssignedMasks', cams=data_config['cams']),
    dict(type='Collect3D',
         keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_masks', 'gt_mask_labels', 'gt_depth', 'gt_assigned_masks'],
         meta_keys=meta_keys)
]
test_pipeline = [
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1280, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img'], meta_keys=meta_keys)
        ])
]

train_ann_file = 'train/front/det_3d.json'
val_ann_file = 'val/front/det_3d.json'

train_img_prefix = 'train/front/img.zip'
val_img_prefix = 'val/front/img.zip'

train_insseg_ann_file = 'train/front/det_insseg_2d.json'
val_insseg_ann_file = 'val/front/det_insseg_2d.json'

train_depth_prefix = 'train/front/depth.zip'
val_depth_prefix = 'val/front/depth.zip'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_file,
        img_prefix=train_img_prefix,
        insseg_ann_file=train_insseg_ann_file,
        depth_prefix=train_depth_prefix,
        backend_type='zip',
        img_to_float32=True,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann_file,
        img_prefix=val_img_prefix,
        insseg_ann_file=val_insseg_ann_file,
        depth_prefix=val_depth_prefix,
        backend_type='zip',
        img_to_float32=True,
        pipeline=test_pipeline,
        test_mode=True,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann_file,
        img_prefix=val_img_prefix,
        insseg_ann_file=val_insseg_ann_file,
        depth_prefix=val_depth_prefix,
        backend_type='zip',
        img_to_float32=True,
        pipeline=test_pipeline,
        test_mode=True,
    ))

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 10
evaluation = dict(interval=10000, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

load_from = '<YOUR_CHECKPOINT_DIR>/detr3d_resnet101.pth'

checkpoint_config = dict(interval=5000, by_epoch=False)
