import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet  # Assuming this is your model implementation
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr, plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
import torch

seed = 42

# General reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# GPU training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def temperature_scaling(logits, temperature):
    temperature = torch.tensor(temperature).unsqueeze(0).unsqueeze(1).expand(logits.size(0), logits.size(1))    #temperature = temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
    scaled_logits = torch.exp(logits / temperature)
    normalization_term = torch.sum(torch.exp(logits / temperature), dim=1, keepdim=True)
    return scaled_logits / normalization_term


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
    parser.add_argument('--subset', default="val")  # Can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    temperatures = [1.0, 0.5, 0.75, 1.1]  # Different temperatures to evaluate

    for temp in temperatures:
        print(f"Processing with temperature T={temp}")
        
        anomaly_score_list = []
        ood_gts_list = []

        modelpath = osp.join(args.loadDir, args.loadModel)
        weightspath = osp.join(args.loadDir, args.loadWeights)

        print("Loading model:", modelpath)
        print("Loading weights:", weightspath)

        model = ERFNet(NUM_CLASSES)

        if not args.cpu:
            model = torch.nn.DataParallel(model).cuda()

        def load_my_state_dict(model, state_dict):  
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

        for path in glob.glob(os.path.expanduser(str(args.input[0]))):
            #print("Processing image:", path)
            images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float()
            images = images.permute(0, 3, 1, 2)
            
            with torch.no_grad():
                logits = model(images)
                logits_scaled = temperature_scaling(logits, temp)
                anomaly_result = 1.0 - np.max(logits_scaled.squeeze(0).data.cpu().numpy(), axis=0)
            
            pathGT = path.replace("images", "labels_masks")
            if "RoadObsticle21" in pathGT:
                pathGT = pathGT.replace("webp", "png")
            if "fs_static" in pathGT:
                pathGT = pathGT.replace("jpg", "png")
            if "RoadAnomaly" in pathGT:
                pathGT = pathGT.replace("jpg", "png")

            mask = Image.open(pathGT)
            ood_gts = np.array(mask)

            # Adjust ood_gts based on specific conditions as in your original code

            if 1 not in np.unique(ood_gts):
                continue
            else:
                ood_gts_list.append(ood_gts)
                anomaly_score_list.append(anomaly_result)

            del logits, logits_scaled, anomaly_result, ood_gts, mask
            torch.cuda.empty_cache()

        ood_gts = np.array(ood_gts_list)
        anomaly_scores = np.array(anomaly_score_list)

        ood_mask = (ood_gts == 1)
        ind_mask = (ood_gts == 0)

        ood_out = anomaly_scores[ood_mask]
        ind_out = anomaly_scores[ind_mask]

        ood_label = np.ones(len(ood_out))
        ind_label = np.zeros(len(ind_out))

        val_out = np.concatenate((ind_out, ood_out))
        val_label = np.concatenate((ind_label, ood_label))

        if len(ood_gts_list) > 0 and len(anomaly_score_list) > 0:
            prc_auc = average_precision_score(val_label, val_out)
            fpr = fpr_at_95_tpr(val_out, val_label)

            print(f'AUPRC score: {prc_auc * 100.0}')
            print(f'FPR@TPR95: {fpr * 100.0}')

            with open('results.txt', 'a') as file:
                file.write(f'Temperature T={temp}:\n')
                file.write(f'    AUPRC score: {prc_auc * 100.0}\n')
                file.write(f'    FPR@TPR95: {fpr * 100.0}\n')
        else:
            print("Source arrays are empty")

        print("\n")

if __name__ == '__main__':
    main()
