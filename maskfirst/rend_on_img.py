import os, sys, cv2, time
from glob import glob
from predictor import COCODemo
from maskrcnn_benchmark.config import cfg


config_file = 'run/CenterMask-R-50-FPN-1x/new_config.yml'
cfg.merge_from_file(config_file)
cfg.merge_from_list(['MODEL.WEIGHT', 'run/CenterMask-R-50-FPN-1x/model_final.pth'])
# cfg.MODEL.WEIGHT = args.weights
cfg.freeze()

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
    composite = coco_demo.run_on_opencv_image(img)
    print("{}\tinference time: {:.2f}s".format(im_name, time.time() - start_time))
    save_path = os.path.join(output_dir, f'{im_name}_result.jpg')
    cv2.imwrite(save_path, composite)