#some comments
# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import random

import torch.nn.functional as F
from PIL import Image
import numpy as np
from erfnet import ERFNet
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr, plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score

# Import the ModelWithTemperature class from the provided code
from temperature_scaling import ModelWithTemperature

seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  # can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    #parser.add_argument('--temperature', type=float, default=0.5)
    args = parser.parse_args()
    anomaly_score_list = []
    ood_gts_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print("Loading model: " + modelpath)
    print("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)

    if not args.cpu:
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print("Model and weights LOADED successfully")
    model.eval()

    temperatures = [1.0 , 0.5, 0.75, 1.1]
    

    for temp in temperatures:
        temp_model = ModelWithTemperature(model, temperature=temp)

    # Function to apply softmax with temperature scaling
    # def softmax_with_temperature(logits, temperature):
        # return F.softmax(logits / temperature, dim=1)

        for path in glob.glob(os.path.expanduser(str(args.input[0]))):
            # print(path)
            images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float()
            images = images.permute(0, 3, 1, 2)
            with torch.no_grad():
                result = temp_model(images)
                # result = model(images)
                # Apply softmax with temperature scaling
                # result_softmax = softmax_with_temperature(result, args.temperature)

            # Calculate MSP anomaly score
            anomaly_result_msp = 1.0 - np.max(result.squeeze(0).data.cpu().numpy(), axis=0)
            
            pathGT = path.replace("images", "labels_masks")                
            if "RoadObsticle21" in pathGT:
                pathGT = pathGT.replace("webp", "png")
            if "fs_static" in pathGT:
                pathGT = pathGT.replace("jpg", "png")                
            if "RoadAnomaly" in pathGT:
                pathGT = pathGT.replace("jpg", "png")  

            mask = Image.open(pathGT)
            ood_gts = np.array(mask)

            if "RoadAnomaly" in pathGT:
                ood_gts = np.where((ood_gts == 2), 1, ood_gts)
            if "LostAndFound" in pathGT:
                ood_gts = np.where((ood_gts == 0), 255, ood_gts)
                ood_gts = np.where((ood_gts == 1), 0, ood_gts)
                ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)

            if "Streethazard" in pathGT:
                ood_gts = np.where((ood_gts == 14), 255, ood_gts)
                ood_gts = np.where((ood_gts < 20), 0, ood_gts)
                ood_gts = np.where((ood_gts == 255), 1, ood_gts)

            if 1 not in np.unique(ood_gts):
                continue
            else:
                ood_gts_list.append(ood_gts)
                anomaly_score_list.append(anomaly_result_msp)

            del result, anomaly_result_msp, ood_gts, mask
            torch.cuda.empty_cache()

        file.write("\n")

        ood_gts = np.array(ood_gts_list)
        anomaly_scores = np.array(anomaly_score_list)

        ood_mask = (ood_gts == 1)
        ind_mask = (ood_gts == 0)

        ood_out_msp = anomaly_scores[ood_mask]
        ind_out_msp = anomaly_scores[ind_mask]

        ood_label = np.ones(len(ood_out_msp))
        ind_label = np.zeros(len(ind_out_msp))

        val_out_msp = np.concatenate((ind_out_msp, ood_out_msp))
        val_label = np.concatenate((ind_label, ood_label))

        if len(ood_gts_list) > 0 and len(anomaly_score_list) > 0:
            if np.any(ood_mask) and np.any(ind_mask):
                prc_auc_msp = average_precision_score(val_label, val_out_msp)
                fpr_msp = fpr_at_95_tpr(val_out_msp, val_label)

                print(f'Temperature: {temp} - AUPRC: {prc_auc_msp * 100.0}, FPR@TPR95: {fpr_msp * 100.0}')


                file.write(f'MSP AUPRC score: {prc_auc_msp * 100.0}, MSP FPR@TPR95: {fpr_msp * 100.0}\n')
            else:
                print("No elements selected by masks")
        else:
            print("Source arrays are empty")
            print(ood_gts)
            print(anomaly_scores)

    file.close()

if __name__ == '__main__':
    main()
