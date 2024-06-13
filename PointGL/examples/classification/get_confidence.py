import os, csv, numpy as np
from tqdm import tqdm
import torch, torch.nn as nn
import shutil
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter
from openpoints.dataset import build_dataloader_from_cfg
from openpoints.models import build_model_from_cfg
from openpoints.models.layers import furthest_point_sample

def predict_probabilities(model, data_loader, cfg):
    model.eval()  # set model to eval mode
    probabilities = []
    paths = []
    for data in tqdm(data_loader, total=len(data_loader)):
        with torch.no_grad():
            for key in data.keys() - ['path']:
                data[key] = data[key].cuda(non_blocking=True)
            points = data['pos']
            data['pos'] = points[:, :, :3].contiguous()
            data['x'] = points[:, :, :cfg.model.encoder_args.in_channels].transpose(1, 2).contiguous()
            logits = model(data)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            paths.extend(data['path'])
            
    return np.concatenate(probabilities, axis=0), paths

def main(gpu, cfg, profile=False):

    eval_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                           cfg.dataset,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split='val',
                                           distributed=cfg.distributed,
                                           )
    eval_loader.dataset.return_path = True
    
    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    model.load_state_dict(torch.load(cfg.pretrained_model_path)["model"])
    model.cuda()
    
    # Predict probabilities
    probabilities, paths = predict_probabilities(model, eval_loader, cfg)    

    # Write probabilities to CSV
    output_csv_path = 'predicted_probabilities.csv'
    with open(output_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Sample', 'Class_0_Probability', 'Class_1_Probability'])
        for i, prob in enumerate(probabilities):
            csv_writer.writerow([{paths[i]}, prob[0], prob[1]])
    print(f"Predicted probabilities saved to {output_csv_path}")

    # Find top n samples for each class starting with "Pred"
    n_samples = 10
    indices_pred = [i for i, path in enumerate(paths) if os.path.basename(path).startswith("Pred")]
    top_class_0 = np.argsort(probabilities[indices_pred, 0])[-n_samples:][::-1]  # Indices of top 5 class 0 samples
    top_class_1 = np.argsort(probabilities[indices_pred, 1])[-n_samples:][::-1]  # Indices of top 5 class 1 samples
    
    # Create folders for top samples
    bad_folder = 'bad_tree_samples'
    good_folder = 'good_tree_samples'

    # Remove existing contents of the folders
    shutil.rmtree(bad_folder, ignore_errors=True)
    shutil.rmtree(good_folder, ignore_errors=True)

    os.makedirs(bad_folder, exist_ok=True)
    os.makedirs(good_folder, exist_ok=True)
    las_path = os.path.join(cfg.dataset.common.data_dir, 'Pred')
    
    # Copy top samples to corresponding folders
    for i, idx in enumerate(top_class_0):
        file_path = os.path.join(las_path, paths[indices_pred[idx]])
        filename = os.path.basename(file_path)  # Extracting original filename
        shutil.copy(file_path, os.path.join(bad_folder, filename))
    for i, idx in enumerate(top_class_1):
        file_path = os.path.join(las_path, paths[indices_pred[idx]])
        filename = os.path.basename(file_path)  # Extracting original filename
        shutil.copy(file_path, os.path.join(good_folder, filename))
        
    print(f"Top {n_samples} samples for class 0 starting with 'Pred' saved to {bad_folder}")
    print(f"Top {n_samples} samples for class 1 starting with 'Pred' saved to {good_folder}")
