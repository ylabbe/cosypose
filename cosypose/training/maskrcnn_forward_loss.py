from cosypose.config import DEBUG_DATA_DIR
import torch

def cast(obj):
    return obj.cuda(non_blocking=True)


def h_maskrcnn(data, model, meters, cfg):
    images, targets = data
    images = list(cast(image).permute(2, 0, 1).float() / 255 for image in images)
    targets = [{k: cast(v) for k, v in t.items()} for t in targets]

    loss_dict = model(images, targets)

    loss_rpn_box_reg = loss_dict['loss_rpn_box_reg']
    loss_objectness = loss_dict['loss_objectness']
    loss_box_reg = loss_dict['loss_box_reg']
    loss_classifier = loss_dict['loss_classifier']
    loss_mask = loss_dict['loss_mask']

    loss = cfg.rpn_box_reg_alpha * loss_rpn_box_reg + \
        cfg.objectness_alpha * loss_objectness + \
        cfg.box_reg_alpha * loss_box_reg + \
        cfg.classifier_alpha * loss_classifier + \
        cfg.mask_alpha * loss_mask

    # torch.save(images, DEBUG_DATA_DIR / 'images.pth.tar')

    meters['loss_rpn_box_reg'].add(loss_rpn_box_reg.item())
    meters['loss_objectness'].add(loss_objectness.item())
    meters['loss_box_reg'].add(loss_box_reg.item())
    meters['loss_classifier'].add(loss_classifier.item())
    meters['loss_mask'].add(loss_mask.item())
    return loss
