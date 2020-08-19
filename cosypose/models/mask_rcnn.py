from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator


class DetectorMaskRCNN(MaskRCNN):
    def __init__(self, input_resize=(240, 320), n_classes=2,
                 backbone_str='resnet50-fpn',
                 anchor_sizes=((32, ), (64, ), (128, ), (256, ), (512, ))):

        assert backbone_str == 'resnet50-fpn'
        backbone = resnet_fpn_backbone('resnet50', pretrained=False)

        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        super().__init__(backbone=backbone, num_classes=n_classes,
                         rpn_anchor_generator=rpn_anchor_generator,
                         max_size=max(input_resize), min_size=min(input_resize))
