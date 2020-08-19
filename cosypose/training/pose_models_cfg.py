# Backbones
from cosypose.models.efficientnet import EfficientNet
from cosypose.models.wide_resnet import WideResNet18, WideResNet34
from cosypose.models.flownet import flownet_pretrained

# Pose models
from cosypose.models.pose import PosePredictor

from cosypose.utils.logging import get_logger
logger = get_logger(__name__)


def check_update_config(config):
    if not hasattr(config, 'init_method'):
        config.init_method = 'v0'
    return config


def create_model_pose(cfg, renderer, mesh_db):
    n_inputs = 6
    backbone_str = cfg.backbone_str
    if backbone_str == 'efficientnet-b3':
        backbone = EfficientNet.from_name('efficientnet-b3', in_channels=n_inputs)
        backbone.n_features = 1536
    elif backbone_str == 'flownet':
        backbone = flownet_pretrained(n_inputs=n_inputs)
        backbone.n_features = 1024
    elif 'resnet34' in backbone_str:
        backbone = WideResNet34(n_inputs=n_inputs)
    elif 'resnet18' in backbone_str:
        backbone = WideResNet18(n_inputs=n_inputs)
    else:
        raise ValueError('Unknown backbone', backbone_str)

    pose_dim = cfg.n_pose_dims

    logger.info(f'Backbone: {backbone_str}')
    backbone.n_inputs = n_inputs
    render_size = (240, 320)
    model = PosePredictor(backbone=backbone,
                          renderer=renderer,
                          mesh_db=mesh_db,
                          render_size=render_size,
                          pose_dim=pose_dim)
    return model


def create_model_refiner(cfg, renderer, mesh_db):
    return create_model_pose(cfg, renderer, mesh_db)


def create_model_coarse(cfg, renderer, mesh_db):
    return create_model_pose(cfg, renderer, mesh_db)
