from cosypose.models.mask_rcnn import DetectorMaskRCNN
from cosypose.utils.logging import get_logger

logger = get_logger(__name__)


def check_update_config(cfg):
    return cfg


def create_model_detector(cfg, n_classes):
    model = DetectorMaskRCNN(input_resize=cfg.input_resize,
                             n_classes=n_classes,
                             backbone_str=cfg.backbone_str,
                             anchor_sizes=cfg.anchor_sizes)
    return model
