import torch
from maskrcnn_benchmark.config import cfg
import argparse, logging, time, datetime
import glob, os
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.utils.logger import setup_logger
import numpy as np


def get_neat_inference_result(coco_eval):
    def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
        p = coco_eval.params
        # import pdb; pdb.set_trace()
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap==1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]   
            # [p.iouThrs * p.recThrs * p.catIds * p.areaRng * p.maxDets]
            s = coco_eval.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,:,aind,mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = coco_eval.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,aind,mind]
        if len(s[s>-1])==0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s>-1])
        print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
        summaryStr = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, summaryStr
    def _summarizeDets():
        stats = np.zeros((12,))
        summaryStrs = [None] * 12
        stats[0], summaryStrs[0] = _summarize(1)
        stats[1], summaryStrs[1] = _summarize(1, iouThr=.5, maxDets=coco_eval.params.maxDets[2])
        stats[2], summaryStrs[2] = _summarize(1, iouThr=.75, maxDets=coco_eval.params.maxDets[2])
        stats[3], summaryStrs[3] = _summarize(1, areaRng='small', maxDets=coco_eval.params.maxDets[2])
        stats[4], summaryStrs[4] = _summarize(1, areaRng='medium', maxDets=coco_eval.params.maxDets[2])
        stats[5], summaryStrs[5] = _summarize(1, areaRng='large', maxDets=coco_eval.params.maxDets[2])
        stats[6], summaryStrs[6] = _summarize(0, maxDets=coco_eval.params.maxDets[0])
        stats[7], summaryStrs[7] = _summarize(0, maxDets=coco_eval.params.maxDets[1])
        stats[8], summaryStrs[8] = _summarize(0, maxDets=coco_eval.params.maxDets[2])
        stats[9], summaryStrs[9] = _summarize(0, areaRng='small', maxDets=coco_eval.params.maxDets[2])
        stats[10], summaryStrs[10] = _summarize(0, areaRng='medium', maxDets=coco_eval.params.maxDets[2])
        stats[11], summaryStrs[11] = _summarize(0, areaRng='large', maxDets=coco_eval.params.maxDets[2])
        return stats, summaryStrs
 
    stats, summaryStrs = _summarizeDets()
    return summaryStrs

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--run-dir",
        default="run/CenterMask-R-50-FPN-1x",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    args = parser.parse_args()

    # import pdb; pdb.set_trace()
    target_dir = args.run_dir
    dir_files = sorted(glob.glob(target_dir+'/*'))
    assert (target_dir + '/new_config.yml') in dir_files, "Error! No cfg file found! check if the dir is right."
    cfg_file = target_dir + '/new_config.yml' if (target_dir + '/new_config.yml') in dir_files else None
    model_files = [f for f in dir_files if f.endswith('00.pth') and 'model_' in f]
    tidyed_before = (target_dir + '/run_res_tidy') in dir_files
    if tidyed_before:
        import pdb; pdb.set_trace()
        pass
    else:
        os.makedirs(target_dir + '/run_res_tidy')

    cfg.merge_from_file(cfg_file)
    cfg.freeze()

    logger = setup_logger("maskrcnn_benchmark", target_dir + '/run_res_tidy', 0, filename="test_log.txt")
    logger.info(cfg)

    # test_str = ''

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=target_dir+'/run_res_tidy/')

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    # output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    # if cfg.OUTPUT_DIR:
    #     for idx, dataset_name in enumerate(dataset_names):
    #         output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
    #         mkdir(output_folder)
    #         output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=False)
    dataset_name = dataset_names[0]
    data_loader_val = data_loaders_val[0]

    for i, model_f in enumerate(model_files):
        # import pdb; pdb.set_trace()
        _ = checkpointer.load(model_f)
        output_folder = target_dir+'/run_res_tidy/'+dataset_name+'_'+(model_f.split('/')[-1][:-4])
        os.makedirs(output_folder)
        logger.info('Processing {}/{}: {}'.format(i, len(model_f), output_folder))
        # print('Processing {}/{}: {}'.format(i, len(model_f), output_folder))
        inference_result = inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        summaryStrs = get_neat_inference_result(inference_result[2][0])
        # test_str += '\n'+ output_folder.split('/')[-1]+   \
        #     '\n'.join(summaryStrs)
        logger.info(output_folder.split('/')[-1])
        logger.info('\n'.join(summaryStrs))

    # with open(target_dir + '/run_res_tidy/test_log.txt', 'w') as f:
    #     f.write(test_str)

if __name__ == "__main__":
    main()
