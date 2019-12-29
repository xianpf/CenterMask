import os, sys, cv2, time, torch
from glob import glob
# from predictor import COCODemo
from maskrcnn_benchmark.config import cfg


config_file = 'run/try_2/new_config.yml'
cfg.merge_from_file(config_file)
cfg.merge_from_list(['MODEL.WEIGHT', 'run/try_2/model_0010000.pth'])
# cfg.MODEL.WEIGHT = args.weights
cfg.freeze()

from maskfirst.do_train_net import MaskFirst
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
# from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from torchvision import transforms as T
from torchvision.transforms import functional as F_transform
from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.colormap import colormap, COLORS
import numpy as np
import matplotlib.pyplot as plt
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints
from maskrcnn_benchmark.utils import cv2_util
from maskrcnn_benchmark.structures.bounding_box import BoxList
import torch.nn.functional as F



class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image):
        size = self.get_size(image.size)
        image = F_transform.resize(image, size)
        return image


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


def expand_masks(mask, padding):
    N = mask.shape[0]
    M = mask.shape[-1]
    pad2 = 2 * padding
    scale = float(M + pad2) / M
    padded_mask = mask.new_zeros((N, 1, M + pad2, M + pad2))
    padded_mask[:, :, padding:-padding, padding:-padding] = mask
    return padded_mask, scale


def expand_masks2(mask, padding):
    N = mask.shape[0]
    M = mask.shape[-1]
    M2 = mask.shape[-2]
    pad2 = 2 * padding
    scale = float(M + pad2) / M
    padded_mask = mask.new_zeros((N, 1, M2 + pad2, M + pad2))
    padded_mask[:, :, padding:-padding, padding:-padding] = mask
    return padded_mask, scale


def paste_mask_in_image(mask, box, im_h, im_w, thresh=0.5, padding=1):
    import pdb; pdb.set_trace()
    # padded_mask, scale = expand_masks(mask[None], padding=padding)
    padded_mask, scale = expand_masks2(mask[None], padding=padding)
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
            result = self.forward_single_image(mask, box)
            results.append(result)
        return results


def vis_keypoints(img, kps, kp_thresh=2, alpha=0.7):
    """Visualizes keypoints (adapted from vis_one_image).
    kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
    """
    dataset_keypoints = PersonKeypoints.NAMES
    kp_lines = PersonKeypoints.CONNECTIONS

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw mid shoulder / mid hip first for better visualization.
    mid_shoulder = (
        kps[:2, dataset_keypoints.index('right_shoulder')] +
        kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
    sc_mid_shoulder = np.minimum(
        kps[2, dataset_keypoints.index('right_shoulder')],
        kps[2, dataset_keypoints.index('left_shoulder')])
    mid_hip = (
        kps[:2, dataset_keypoints.index('right_hip')] +
        kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
    sc_mid_hip = np.minimum(
        kps[2, dataset_keypoints.index('right_hip')],
        kps[2, dataset_keypoints.index('left_hip')])
    nose_idx = dataset_keypoints.index('nose')
    if sc_mid_shoulder > kp_thresh and kps[2, nose_idx] > kp_thresh:
        cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(kps[:2, nose_idx]),
            color=colors[len(kp_lines)], thickness=2, lineType=cv2.LINE_AA)
    if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
        cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(mid_hip),
            color=colors[len(kp_lines) + 1], thickness=2, lineType=cv2.LINE_AA)

    # Draw the keypoints.
    for l in range(len(kp_lines)):
        i1 = kp_lines[l][0]
        i2 = kp_lines[l][1]
        p1 = kps[0, i1], kps[1, i1]
        p2 = kps[0, i2], kps[1, i2]
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)



class COCODemo(object):
    # COCO categories for pretty print
    CATEGORIES = [
        "__background",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    def __init__(
        self,
        cfg,
        confidence_threshold,
        show_mask_heatmaps=False,
        masks_per_dim=2,
        min_image_size=224,
        display_text = False,
        display_scores = False,
    ):
        self.cfg = cfg.clone()
        # self.model = build_detection_model(cfg)
        self.model = MaskFirst(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size

        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()
        # self.transforms = build_transforms(cfg, is_train=False)

        mask_threshold = -1 if show_mask_heatmaps else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        # self.confidence_thresholds_for_classes = torch.tensor(confidence_thresholds_for_classes)
        self.confidence_threshold = confidence_threshold
        self.show_mask_heatmaps = show_mask_heatmaps
        self.masks_per_dim = masks_per_dim

        self.display_score = display_scores
        self.display_text = display_text

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        transform = T.Compose(
            [
                T.ToPILImage(),
                Resize(min_size, max_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def run_on_opencv_image(self, image):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        predictions = self.compute_prediction(image)
        top_predictions = self.select_top_predictions(predictions)

        result = image.copy()
        if self.show_mask_heatmaps:
            return self.create_mask_montage(result, top_predictions)
        result = self.overlay_boxes(result, top_predictions)
        if self.cfg.MODEL.MASK_ON:
            result = self.overlay_mask(result, top_predictions)
        if self.cfg.MODEL.KEYPOINT_ON:
            result = self.overlay_keypoints(result, top_predictions)
        if self.display_text:
            result = self.overlay_class_names(result, top_predictions, self.display_score)

        return result

    def compute_prediction(self, original_image):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image
        image = self.transforms(original_image)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list)
        import pdb; pdb.set_trace()
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            import pdb; pdb.set_trace()
            # always single image is passed at a time
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)
        return prediction

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        if predictions.has_field("mask_scores"):
            scores = predictions.get_field("mask_scores")
        else:
            scores = predictions.get_field("scores")
        labels = predictions.get_field("labels")
        # thresholds = self.confidence_thresholds_for_classes[(labels - 1).long()]
        # keep = torch.nonzero(scores > thresholds).squeeze(1)
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def compute_colors_for_labels_yolact(self, labels, class_color=False):
        """
        Simple function that adds fixed colors depending on the class
        """
        # colors = labels[:, None] * self.palette
        # colors = (colors % 255).numpy().astype("uint8")
        # colors = torch.cat([(class * 5) % len(COLORS) for class in labels])
        color_indice = [(labels[c] * 5) if class_color else c * 5 % len(COLORS) for c in range(len(labels))]
        # colors = torch.cat([COLORS[color_idx] for color_idx in color_indice])
        colors = [COLORS[color_idx] for color_idx in color_indice]
        return colors

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        # colors = self.compute_colors_for_labels(labels).tolist()
        colors = colormap(rgb=True).tolist()
        # colors = self.compute_colors_for_labels_yolact(labels)


        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 1
            )

        return image

    def overlay_mask(self, image, predictions):
        """
        Adds the instances FCOS-MASK-SCORING-PLUS-R-50-FPN-MaskP3P5-new2xcontours for each predicted object.
        Each label has a different color.

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        masks = predictions.get_field("mask").numpy()
        labels = predictions.get_field("labels")

        #original
        # colors = self.compute_colors_for_labels(labels).tolist()

        #Detectron.pytorch for matplotlib colors
        colors = colormap(rgb=True).tolist()
        # colors = self.compute_colors_for_labels_yolact(labels)

        
        mask_img = np.copy(image)


        for mask, color in zip(masks, colors):
            # color_mask = color_list[color_id % len(color_list)]
            # color_id += 1

            # thresh = mask[0, :, :, None]
            thresh = mask[0, :, :, None].astype(np.uint8)
            contours, hierarchy = cv2_util.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            mask_img = cv2.drawContours(mask_img, contours, -1, color, -1)


        # composite = image
        alpha = 0.45
        composite = cv2.addWeighted(image, 1.0 - alpha, mask_img, alpha, 0)
        # composite = cv2.addWeighted(image, 1.0 - alpha, mask, alpha, 0)

        return composite

    def overlay_keypoints(self, image, predictions):
        keypoints = predictions.get_field("keypoints")
        kps = keypoints.keypoints
        scores = keypoints.get_field("logits")
        kps = torch.cat((kps[:, :, 0:2], scores[:, :, None]), dim=2).numpy()
        for region in kps:
            image = vis_keypoints(image, region.transpose((1, 0)))
        return image

    def create_mask_montage(self, image, predictions):
        """
        Create a montage showing the probability heatmaps for each one one of the
        detected objects

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask`.
        """
        masks = predictions.get_field("mask")
        masks_per_dim = self.masks_per_dim
        masks = L.interpolate(
            masks.float(), scale_factor=1 / masks_per_dim
        ).byte()
        height, width = masks.shape[-2:]
        max_masks = masks_per_dim ** 2
        masks = masks[:max_masks]
        # handle case where we have less detections than max_masks
        if len(masks) < max_masks:
            masks_padded = torch.zeros(max_masks, 1, height, width, dtype=torch.uint8)
            masks_padded[: len(masks)] = masks
            masks = masks_padded
        masks = masks.reshape(masks_per_dim, masks_per_dim, height, width)
        result = torch.zeros(
            (masks_per_dim * height, masks_per_dim * width), dtype=torch.uint8
        )
        for y in range(masks_per_dim):
            start_y = y * height
            end_y = (y + 1) * height
            for x in range(masks_per_dim):
                start_x = x * width
                end_x = (x + 1) * width
                result[start_y:end_y, start_x:end_x] = masks[y, x]
        return cv2.applyColorMap(result.numpy(), cv2.COLORMAP_JET)

    def overlay_class_names(self, image, predictions, display_score=False):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        # labels = predictions.get_field("labels").tolist()
        labels = predictions.get_field("labels")
        # colors = self.compute_colors_for_labels(labels).tolist()
        colors = colormap(rgb=True).tolist()
        # colors = self.compute_colors_for_labels_yolact(labels)

        labels = labels.tolist()
        labels = [self.CATEGORIES[i] for i in labels]
        boxes = predictions.bbox


        # font_face = cv2.FONT_HERSHEY_COMPLEX
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        for box, score, label, color in zip(boxes, scores, labels, colors):
            x, y = box[:2]
            if display_score:
                template = "{}: {:.2f}"
                s = template.format(label, score)
            else:
                s = label
            text_w, text_h = cv2.getTextSize(s, font_face, font_scale, font_thickness)[0]
            text_pt = (x, y - 3)
            text_color = [255, 255, 255]
            # cv2.rectangle(image, (x,y), (x + text_w, y - text_h - 2), color,  -1)
            # cv2.putText(image, s, (x, y), font_face, font_scale, (255, 255, 255), 1)
            cv2.rectangle(image, (x,y), (x + text_w, y - text_h - 4), color,  -1) # mimicing yolact
            cv2.putText(image, s, text_pt, font_face, font_scale, text_color,font_thickness,  cv2.LINE_AA)

        return image

def one_shot_plot(original_image):
    # apply pre-processing to image
    model = MaskFirst(cfg)
    model.eval()
    device = torch.device(cfg.MODEL.DEVICE)
    cpu_device = torch.device("cpu")
    model.to(device)
    masker = Masker(threshold=0.2, padding=1)

    # image = self.transforms(original_image)
    if cfg.INPUT.TO_BGR255:
        to_bgr_transform = T.Lambda(lambda x: x * 255)
    else:
        to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
    )
    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    transform = T.Compose(
        [
            T.ToPILImage(),
            Resize(min_size, max_size),
            T.ToTensor(),
            to_bgr_transform,
            normalize_transform,
        ]
    )
    image = transform(original_image)

    # convert to an ImageList, padded so that it is divisible by
    # cfg.DATALOADER.SIZE_DIVISIBILITY
    image_list = to_image_list(image, cfg.DATALOADER.SIZE_DIVISIBILITY)
    image_list = image_list.to(device)
    # compute predictions
    with torch.no_grad():
        masks_list = model(image_list)
    masks = masks_list[0].to(cpu_device)
    # import pdb; pdb.set_trace()
    height, width = original_image.shape[:-1]
    masks_scale = F.interpolate(masks, (height, width), mode='bilinear', align_corners=False)
    masks_bool = masks_scale > 0.2

    fake_box = torch.tensor([[100, 50, 200, 150]]*len(masks_bool), device=masks_bool.device)
    box_fake = BoxList(fake_box, tuple(masks_bool.shape[-2:]), mode="xyxy")
    box_fake.add_field('mask', masks_bool)
    keep = masks_scale.view(len(masks_scale), -1).max(dim=1)

    # top_predictions = self.select_top_predictions(predictions)
    import pdb; pdb.set_trace()

    result = original_image.copy()
    # result = self.overlay_mask(result, box_fake)
    masks_np = box_fake.get_field("mask").numpy()

    colors = colormap(rgb=True).tolist()

    mask_img = np.copy(original_image)

    # import pdb; pdb.set_trace()
    # for mask, color in zip(masks_np, colors):
    for mask, color in zip(masks_np[:3], colors):
        # color_mask = color_list[color_id % len(color_list)]
        # color_id += 1

        # thresh = mask[0, :, :, None]
        thresh = mask[0, :, :, None].astype(np.uint8)
        contours, hierarchy = cv2_util.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        mask_img = cv2.drawContours(mask_img, contours, -1, color, -1)


    # import pdb; pdb.set_trace()
    # composite = image
    alpha = 0.45
    composite = cv2.addWeighted(original_image, 1.0 - alpha, mask_img, alpha, 0)
    return composite


    # compute predictions
        # with torch.no_grad():
        #     predictions = model(image_list)
        # predictions = [o.to(cpu_device) for o in predictions]

        # # always single image is passed at a time
        # prediction = predictions[0]

        # # reshape prediction (a BoxList) into the original image size
        # height, width = original_image.shape[:-1]
        # prediction = prediction.resize((width, height))

        # if prediction.has_field("mask"):
        #     # if we have masks, paste the masks in the right position
        #     # in the image, as defined by the bounding boxes
        #     masks = prediction.get_field("mask")
        #     # import pdb; pdb.set_trace()
        #     # always single image is passed at a time
        #     masks = masker([masks], [prediction])[0]
        #     prediction.add_field("mask", masks)
        # return prediction

coco_demo = COCODemo(cfg, confidence_threshold=0.2, display_text=True, display_scores=True)

img_path = 'datasets/coco/val2014/COCO_val2014_000000000139.jpg'
output_dir = 'run/develop/show/'

if os.path.isfile(img_path):
    imglist = [img_path]
else:
    imglist = glob(os.path.join(img_path, '*'))

# import pdb; pdb.set_trace()

for i in range(len(imglist)):
    print('file', i)
    img = cv2.imread(imglist[i])
    assert img is not None
    im_name, _ = os.path.splitext(os.path.basename(imglist[i]))
    print(f"{im_name} processing...")
    start_time = time.time()
    composite = one_shot_plot(img)
    # composite = coco_demo.run_on_opencv_image(img)
    print("{}\tinference time: {:.2f}s".format(im_name, time.time() - start_time))
    save_path = os.path.join(output_dir, f'{im_name}_result.jpg')
    cv2.imwrite(save_path, composite)