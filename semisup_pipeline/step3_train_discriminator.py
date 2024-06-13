import os.path as osp
import torch
import torch.nn.parallel
import tqdm
import numpy as np
from TLSpecies.simpleview_pytorch import SimpleView
from TLSpecies.utils.discriminator_dataset import RealPredDataset
from TLSpecies.utils.train import train as train_simpleview
from TLSpecies.utils import plot_depth_images
from torch.utils.data import DataLoader, random_split, Subset
import torch.optim as optim
import torch.nn as nn
import multiprocessing as mp
import copy
import csv



TREE_CLASS_IN_PYTORCH_DATASET = 0
NON_TREES_LABEL_IN_GROUPING = 0
NOT_ASSIGNED_LABEL_IN_GROUPING = -1
START_NUM_PREDS = 1


def train_discriminator(model, train_loader, val_loader, params, device):
    if params['loss_fn']=="cross-entropy":
        loss_fn = nn.CrossEntropyLoss()
        print("Using cross-entropy loss...")
    if params['loss_fn']=="smooth-loss":
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2)
        print("Using smooth-loss")

    if type(params['learning_rate']) == list:
        lr = params['learning_rate'][0]
        step_size = params['learning_rate'][1]
        gamma = params['learning_rate'][2]
    else:
        lr = params['learning_rate']

    if params['optimizer']=="sgd":
        print("Optimizing with SGD...")
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=params['momentum'])
    elif params['optimizer']=="adam":
        print("Optimizing with AdaM...")
        optimizer = optim.Adam(model.parameters(), lr=lr)


    best_acc = 0

    for epoch in range(params['epoch']):  # loop over the dataset multiple times
        #Training loop============================================
        model.train()
        running_loss = 0.0
        
        for i, data in enumerate(train_loader, 0):
            depth_images = data['depth_images']
            labels = data['labels']

            depth_images = depth_images.to(device=device)
            labels = labels.to(device=device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(depth_images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 5 == 4:    # print every 5 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2))
                running_loss = 0.0

        model.eval()  
        num_train_correct = 0
        num_train_samples = 0

        num_val_correct = 0
        num_val_samples = 0

        running_train_loss = 0
        running_val_loss = 0

        with torch.no_grad():
            for data in train_loader:
                depth_images = data['depth_images']
                labels = data['labels']

                depth_images = depth_images.to(device=device)
                labels = labels.to(device=device)

                scores = model(depth_images)
                _, predictions = scores.max(1)
                num_train_correct += (predictions == labels).sum()
                num_train_samples += predictions.size(0)

                running_train_loss += loss_fn(scores, labels)

            train_acc = float(num_train_correct)/float(num_train_samples)
            train_loss = running_train_loss/len(train_loader)

            print(f'OVERALL (Train): Got {num_train_correct} / {num_train_samples} with accuracy {train_acc*100:.2f}')

            #Val set eval===============
            all_labels = torch.tensor([]).to(device)
            all_predictions = torch.tensor([]).to(device)

            for data in val_loader:
                depth_images = data['depth_images']
                labels = data['labels']

                depth_images = depth_images.to(device=device)
                labels = labels.to(device=device)

                scores = model(depth_images)
                _, predictions = scores.max(1)

                all_labels = torch.cat((all_labels, labels))
                all_predictions = torch.cat((all_predictions, predictions))

                num_val_correct += (predictions == labels).sum()
                num_val_samples += predictions.size(0)

                running_val_loss += loss_fn(scores, labels)

            val_acc = float(num_val_correct)/float(num_val_samples)
            val_loss = running_val_loss/len(val_loader)

            print(f'OVERALL (Val): Got {num_val_correct} / {num_val_samples} with accuracy {val_acc*100:.2f}')
                

            if val_acc >= best_acc:
                best_model_state = copy.deepcopy(model.state_dict())
                best_acc = val_acc

    print('Saving best (val) model...')
    print('Best overall accuracy: {}'.format(best_acc))
    torch.save(best_model_state,
               '{model_dir}/best_model.pth'.format(model_dir=params['model_dir'])
              )
    print('Saved!')

    model.load_state_dict(best_model_state)

    return model    

def inspect_results(dataset, loader, model, device, output_path, n_samples=20):
    #logits, labels, predictions, _ = utils.predict_from_dirs('data/trees_test.pt', 'models/2024-04-10 18:06:14.813858_best', params={'species':["Real", "Pred"], 'num_views':6})
    probs = []
    ids = []
    for data in loader:
        depth_images = data['depth_images']
        labels = data['labels']
        id = data['ids']

        depth_images = depth_images.to(device=device)
        labels = labels.to(device=device)

        scores = model(depth_images)
        prob = torch.nn.functional.softmax(scores, dim=1)
        prob = prob.detach().cpu().numpy()
        probs.append(prob)
        ids.append(id)
    probs = np.concatenate(probs, axis=0)

    top_class_0 = np.argsort(probs[:, 1])
    top_class_1 = np.argsort(probs[:, 0])
    for i in range(n_samples):
        sample = dataset[top_class_0[i].item()]
        imgs = sample['depth_images']
        id = sample['ids']
        prob = probs[top_class_0[i].item()]
        fig, ax = plot_depth_images(imgs, title=f"{id} - {prob[0]:.2f} predicted")
        fig.savefig(osp.join(output_path, f"top_{i}.png"))

        sample = dataset.__getitem__(top_class_1[i].item())
        imgs = sample['depth_images']
        id = sample['ids']
        prob = probs[top_class_1[i].item()]
        fig, ax = plot_depth_images(imgs, title=f"{id} - {prob[0]:.2f} predicted")
        fig.savefig(osp.join(output_path, f"bottom_{i}.png"))
    
    output_csv_path = osp.join(output_path, 'predicted_probabilities.csv')
    with open(output_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Sample', 'Class_0_Probability', 'Class_1_Probability'])
        for i in range(len(probs)):
            csv_writer.writerow([ids[i], probs[i][0], probs[i][1]])
    
    print(f"Saved predicted probabilities to {output_csv_path}")


def main():
    mp.set_start_method('spawn')
    torch.cuda.max_memory_allocated(1024*1024*1024*8)

    discriminator_dataset_path = 'data/step2_data_for_discriminator/real_pred_dataset/RealPredDataset.pt'

    discriminator = SimpleView(6, 2).cuda()
    
    discriminator_dataset = torch.load(discriminator_dataset_path)
    total_size = len(discriminator_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_indices, val_indices = random_split(discriminator_dataset, [train_size, val_size])

    # Create data loaders for train and validation sets
    train_dataloader = DataLoader(
        train_indices,
        batch_size=1,
        num_workers=1,
        shuffle=True,
        drop_last=True,
        pin_memory=True)

    val_dataloader = DataLoader(
        val_indices,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        drop_last=False,
        pin_memory=True)


    # train discriminator model
    print("Training discriminator model...")
    discriminator_params = {
        'epoch': 12,
        'learning_rate': 0.005,
        'optimizer': 'adam',
        'momentum': 0.9,
        'loss_fn': 'cross-entropy',
        'model_dir': 'models/discriminator',
    }
    discriminator = train_discriminator(discriminator, train_dataloader, val_dataloader, discriminator_params, 'cuda')
    print("Saving discriminator model...")
    torch.save(discriminator, 'models/discriminator/best_model.pt')

    inspect_results(val_indices, val_dataloader, discriminator, 'cuda', 'data/step2_data_for_discriminator/samples/')


if __name__ == '__main__':
    main()