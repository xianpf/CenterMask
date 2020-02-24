import torch
import torch.nn.functional as F
from maskrcnn_benchmark.structures.bounding_box import BoxList

# the next two functions should be merged inside Masker
# but are kept here for the moment while we need them
# temporarily gor paste_mask_in_image
def expand_boxes(boxes, scale):
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp


def expand_boxes_new(boxes, scale):
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale[0]
    h_half *= scale[1]

    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp


def expand_masks(mask, padding):
    N = mask.shape[0]
    M = mask.shape[-1]
    pad2 = 2 * padding
    scale = float(M + pad2) / M
    padded_mask = mask.new_zeros((N, 1, M + pad2, M + pad2))
    padded_mask[:, :, padding:-padding, padding:-padding] = mask
    return padded_mask, scale


def expand_masks_new(mask, padding):
    N = mask.shape[0]
    W = mask.shape[1]
    H = mask.shape[2]
    pad2 = 2 * padding
    scaleW = float(W + pad2) / W
    scaleH = float(H + pad2) / H
    padded_mask = mask.new_zeros((N, 1, W + pad2, H + pad2))
    padded_mask[:, :, padding:-padding, padding:-padding] = mask
    return padded_mask, (scaleW, scaleH)


def paste_mask_in_image(mask, box, im_h, im_w, thresh=0.5, padding=1):
    padded_mask, scale = expand_masks(mask[None], padding=padding)
    mask = padded_mask[0, 0]
    box = expand_boxes(box[None], scale)[0]
    box = box.to(dtype=torch.int32)

    TO_REMOVE = 1
    w = box[2] - box[0] + TO_REMOVE
    h = box[3] - box[1] + TO_REMOVE
    w = max(w, 1)
    h = max(h, 1)

    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, 1, -1, -1))

    # Resize mask
    mask = mask.to(torch.float32)
    mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
    mask = mask[0][0]

    if thresh >= 0:
        mask = mask > thresh
    else:
        # for visualization and debugging, we also
        # allow it to return an unmodified mask
        # mask = (mask * 255).to(torch.uint8)
        mask = (mask * 255).to(torch.bool)

    # im_mask = torch.zeros((im_h, im_w), dtype=torch.uint8)
    im_mask = torch.zeros((im_h, im_w), dtype=torch.bool)
    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)

    im_mask[y_0:y_1, x_0:x_1] = mask[
        (y_0 - box[1]) : (y_1 - box[1]), (x_0 - box[0]) : (x_1 - box[0])
    ]
    return im_mask



class Masker(object):
    """
    Projects a set of masks in an image on the locations
    specified by the bounding boxes
    """

    def __init__(self, threshold=0.5, padding=1):
        self.threshold = threshold
        self.padding = padding

    def forward_single_image(self, masks, boxes):
        boxes = boxes.convert("xyxy")
        im_w, im_h = boxes.size
        res = [
            paste_mask_in_image(mask[0], box, im_h, im_w, self.threshold, self.padding)
            for mask, box in zip(masks, boxes.bbox)
        ]
        if len(res) > 0:
            res = torch.stack(res, dim=0)[:, None]
        else:
            res = masks.new_empty((0, 1, masks.shape[-2], masks.shape[-1]))
        return res

    def maskpyramid_masker(self, masks, boxes):
        # import pdb; pdb.set_trace()
        mask_h, mask_w = masks.shape[-2:]
        box_w, box_h = boxes.size
        data_w, data_h = tuple(boxes.get_field('dataloader_size').tolist())

        masks_content = masks[:,:,:round(data_h/16), :round(data_w/16)]
        bg_content = boxes.get_field('bg_logits')[:,:,:round(data_h/16), :round(data_w/16)]
        masks_1 = F.interpolate(masks_content, size=(box_h, box_w), mode='bilinear', align_corners=False)
        masks_0 = F.interpolate(bg_content, size=(box_h, box_w), mode='bilinear', align_corners=False)
        res = masks_1 > masks_0
        
        return res

    def __call__(self, masks, boxes):
        if isinstance(boxes, BoxList):
            boxes = [boxes]

        # Make some sanity check
        assert len(boxes) == len(masks), "Masks and boxes should have the same length."

        # TODO:  Is this JIT compatible?
        # If not we should make it compatible.
        results = []
        for mask, box in zip(masks, boxes):
            assert mask.shape[0] == len(box), "Number of objects should be the same."
            if 'bg_logits' in box.extra_fields.keys():
                result = self.maskpyramid_masker(mask, box)
            else:
                result = self.forward_single_image(mask, box)
            # import pdb; pdb.set_trace()
            results.append(result)
        return results
