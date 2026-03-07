# MMAction2 config template: skeleton-based push-up classifier (pseudo labels)
# Usage (inside mmaction2 repo):
#   python tools/train.py /ABS/PATH/TO/this_config.py

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        graph_cfg=dict(layout='coco', mode='spatial'),
        in_channels=3,
    ),
    cls_head=dict(type='GCNHead', num_classes=2, in_channels=256),
    data_preprocessor=dict(type='ActionDataPreprocessor'),
)

dataset_type = 'PoseDataset'
ann_file_train = '/Users/taiwu/Documents/GitHub/yolo-pose-pushup-counter/datasets/mmaction2_pushup/train.pkl'
ann_file_val = '/Users/taiwu/Documents/GitHub/yolo-pose-pushup-counter/datasets/mmaction2_pushup/val.pkl'

train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSampleFrames', clip_len=48),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs'),
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSampleFrames', clip_len=48, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs'),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type=dataset_type, ann_file=ann_file_train, pipeline=train_pipeline),
)
val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type, ann_file=ann_file_val, pipeline=val_pipeline, test_mode=True),
)
test_dataloader = val_dataloader

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=40, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=5e-4),
)
param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=40, eta_min=1e-5, by_epoch=True),
]

default_scope = 'mmaction'
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, save_best='auto'),
    logger=dict(type='LoggerHook', interval=20),
)

work_dir = './work_dirs/pushup_stgcn_pseudo'

# Optional: load from official skeleton pretrain
load_from = None
resume = False
