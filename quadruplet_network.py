"""
Quadruplet Network for Attribute Recognition and Person Re-Identification
University of Trento — Giacomo Lazzerini, Dario Fabiani

Converted from Jupyter notebook to standalone Python script with bug fixes applied.
"""

import os
import random
import sys
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cuda
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

# FIX: set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# FIX: all hyperparameters are now centralised in config (previously scattered
# across the notebook cells and hardcoded inside class initialisers)
config = dict(
    train_path='./data/train',
    test_path='./data/test',
    queries_path='./data/queries',
    csv_file='./data/annotations_train.csv',
    mAP_rank=20,
    image_size=(224, 224),
    batch_size=48,
    epochs=50,
    num_bottleneck=256,
    # FIX: quadruplet-loss margins moved here from QuadrupletLoss.__init__
    margin1=2.0,
    margin2=1.0,
    # FIX: combined-loss weight moved here from OverallWrapper.__init__
    lambda_weight=0.8,
    # FIX: optimiser hyper-params moved here from main() signature defaults
    learning_rate=0.001,
    weight_decay=0.000001,
    momentum=0.5,
    # FIX: make worker count dynamic so it never exceeds the OS limit
    num_workers=min(4, os.cpu_count() or 2),
)

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def build_dataframe(config):
    csv = pd.read_csv(config['csv_file'])
    labels = csv.columns[1:].tolist()

    headers = ['image_name', 'ID', 'cam']
    headers.extend(labels)

    subject_dict = {i: [] for i in headers}

    for file in tqdm(os.listdir(config['train_path'])):
        filename = os.fsdecode(file)
        if filename.endswith('.jpg'):
            ID = int(file.split('_')[0])
            subject_dict['ID'].append(ID)
            cam = int(file.split('_')[1][1])
            subject_dict['cam'].append(cam)
            subject_dict['image_name'].append(filename)
            subject_row = csv.loc[csv['id'] == ID]
            for header in list(subject_row.columns)[1:]:
                subject_dict[header].append(subject_row.iloc[0][header] - 1)

    complete_df = pd.DataFrame(subject_dict, columns=headers)
    complete_df['upmulti'] = np.where(
        complete_df.loc[:, complete_df.columns[13:21]].sum(axis=1) == 0, 1, 0
    )
    complete_df['downmulti'] = np.where(
        complete_df.loc[:, complete_df.columns[21:-1]].sum(axis=1) == 0, 1, 0
    )

    att_map = [1 if x == 2 else x for x in complete_df.nunique()[3:].tolist()]
    return complete_df, att_map


def build_splits(complete_df, config):
    IDs = list(set(list(complete_df['ID'])))
    random.shuffle(IDs)

    training_ids, validation_ids = sklearn.model_selection.train_test_split(
        IDs, test_size=0.28, random_state=42
    )

    train_df = complete_df.loc[complete_df['ID'].isin(training_ids)]
    valid_df = complete_df.loc[complete_df['ID'].isin(validation_ids)]

    val_ID_list = list(valid_df['ID'])
    proportion_test_queries = round(
        len(os.listdir(config['queries_path'])) / len(os.listdir(config['test_path'])), 2
    )

    validation_test_df, validation_queries_df = sklearn.model_selection.train_test_split(
        valid_df,
        test_size=proportion_test_queries,
        stratify=val_ID_list,
        random_state=42,
    )
    validation_test_df.reset_index(drop=True, inplace=True)
    validation_queries_df.reset_index(drop=True, inplace=True)

    return train_df, valid_df, validation_test_df, validation_queries_df


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MarketDataset(Dataset):

    def __init__(self, dataframe, root_dir, transform=None, train_set=True, val_set=False):
        self.annotations = dataframe
        self.root_dir = root_dir
        self.list_dir = sorted(os.listdir(self.root_dir))
        self.transform = transform
        self.train_set = train_set
        self.val_set = val_set

    def __len__(self):
        if self.train_set or self.val_set:
            return self.annotations.shape[0]
        return len(self.list_dir)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.list_dir[index])
        anchor_image = Image.open(img_path)

        if self.train_set:
            attributes = torch.tensor(
                [int(self.annotations.iat[index, i]) for i in range(3, len(self.annotations.columns))]
            )
            raw_ID = self.annotations.iat[index, 1]

            positive_list = list(self.annotations[self.annotations['ID'] == raw_ID]['image_name'])[1:]
            positive_item = random.choice(positive_list)
            positive_image = Image.open(os.path.join(self.root_dir, positive_item))

            negative_list = list(self.annotations[self.annotations['ID'] != raw_ID]['image_name'])[1:]
            negative_items = random.sample(negative_list, 2)
            first_negative_image = Image.open(os.path.join(self.root_dir, negative_items[0]))
            second_negative_image = Image.open(os.path.join(self.root_dir, negative_items[1]))

            if self.transform:
                anchor_image = self.transform(anchor_image)
                positive_image = self.transform(positive_image)
                first_negative_image = self.transform(first_negative_image)
                second_negative_image = self.transform(second_negative_image)

            return anchor_image, attributes, [positive_image, first_negative_image, second_negative_image]
        else:
            if self.transform:
                anchor_image = self.transform(anchor_image)
            return anchor_image


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

jitter_param = 0.4

train_tfms = T.Compose([
    T.Resize(size=config['image_size']),
    T.RandomCrop(32, padding=4, padding_mode='reflect'),
    T.RandomHorizontalFlip(),
    T.ColorJitter(
        brightness=jitter_param,
        contrast=jitter_param,
        saturation=jitter_param,
    ),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

valid_tfms = T.Compose([
    T.Resize(size=config['image_size']),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_datasets(train_df, valid_df, validation_test_df, validation_queries_df, config):
    train_ds = MarketDataset(
        dataframe=train_df, root_dir=config['train_path'],
        transform=train_tfms, train_set=True,
    )
    valid_ds = MarketDataset(
        dataframe=valid_df, root_dir=config['train_path'],
        transform=valid_tfms, train_set=True,
    )

    # datasets used for mAP evaluation during training
    validation_test_ds = MarketDataset(
        dataframe=validation_test_df, root_dir=config['train_path'],
        # FIX: was using train_tfms (with random augmentations) — switched to valid_tfms
        transform=valid_tfms, train_set=False, val_set=True,
    )
    validation_queries_ds = MarketDataset(
        dataframe=validation_queries_df, root_dir=config['train_path'],
        transform=valid_tfms, train_set=False, val_set=True,
    )

    # datasets used for final inference (attribute CSV + Re-ID txt)
    test_ds = MarketDataset(
        dataframe=validation_test_df, root_dir=config['test_path'],
        # FIX: was using train_tfms — test data must not be randomly augmented
        transform=valid_tfms, train_set=False,
    )
    queries_ds = MarketDataset(
        dataframe=validation_queries_df, root_dir=config['queries_path'],
        # FIX: was using train_tfms — test data must not be randomly augmented
        transform=valid_tfms, train_set=False,
    )

    return train_ds, valid_ds, validation_test_ds, validation_queries_ds, test_ds, queries_ds


def build_dataloaders(train_ds, valid_ds, validation_test_ds, validation_queries_ds,
                      test_ds, queries_ds, config):
    nw = config['num_workers']

    train_loader = DataLoader(
        train_ds, config['batch_size'], shuffle=True, num_workers=nw, drop_last=True
    )
    val_loader = DataLoader(
        valid_ds, config['batch_size'], shuffle=False, num_workers=nw
    )

    # FIX: validation_test_loader had shuffle=True — evaluation loaders must be deterministic
    validation_test_loader = DataLoader(
        validation_test_ds, config['batch_size'], shuffle=False, num_workers=nw, drop_last=True
    )
    validation_queries_loader = DataLoader(
        validation_queries_ds, config['batch_size'], shuffle=False, num_workers=nw
    )

    test_loader = DataLoader(
        test_ds, config['batch_size'], shuffle=False, num_workers=nw
    )
    queries_loader = DataLoader(
        queries_ds, config['batch_size'], shuffle=False, num_workers=nw
    )

    return (train_loader, val_loader, validation_test_loader,
            validation_queries_loader, test_loader, queries_loader)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ClassificationBlock(nn.Module):
    def __init__(self, input_dim, class_dim, num_bottleneck=config['num_bottleneck']):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_bottleneck)
        self.fc2 = nn.Linear(num_bottleneck, class_dim)
        self.bns = nn.BatchNorm1d(num_bottleneck)

    def forward(self, x):
        x = self.fc(x)
        x = self.bns(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=0.5)
        x = self.fc2(x)
        return x


class Backbone(nn.Module):
    def __init__(self, attribute_map, model_name='resnet18'):
        super().__init__()
        self.model_name = model_name
        self.class_num = len(attribute_map)
        self.attribute_map = attribute_map

        model_ft = getattr(models, self.model_name)(pretrained=True)
        if self.model_name.lower().startswith('resnet'):
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.fc = nn.Sequential()
            self.features = model_ft
            self.num_ftrs = 512
        elif self.model_name.lower().startswith('densenet'):
            model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.fc = nn.Sequential()
            self.features = model_ft.features
            self.num_ftrs = 1024
        else:
            raise NotImplementedError

        for c in range(self.class_num):
            self.__setattr__(
                'class_%d' % c,
                ClassificationBlock(input_dim=self.num_ftrs, class_dim=attribute_map[c]),
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        raw_pred_label = [self.__getattr__('class_%d' % c)(x) for c in range(self.class_num)]
        prob_pred_label = [
            torch.sigmoid(self.__getattr__('class_%d' % c)(x))
            if self.__getattr__('class_%d' % c)(x).size()[1] == 1
            else torch.softmax(self.__getattr__('class_%d' % c)(x), dim=1)
            for c in range(self.class_num)
        ]

        return raw_pred_label, prob_pred_label, x


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class AttributesLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, attrs):
        bce = nn.BCEWithLogitsLoss()
        cross_entropy = nn.CrossEntropyLoss()
        binary_losses = 0
        cross_losses = 0
        for idx in range(len(preds[0])):
            if preds[0][idx].size()[1] == 1:
                binary_losses += bce(preds[0][idx], attrs[:, idx].unsqueeze(1).to(torch.float32))
            else:
                cross_losses += cross_entropy(preds[0][idx], attrs[:, idx])
        return cross_losses + binary_losses


class QuadrupletLoss(nn.Module):
    """
    Extends triplet loss with a second negative to push negative pairs apart.
    """

    def __init__(self, margin1=config['margin1'], margin2=config['margin2']):
        super().__init__()
        self.margin1 = margin1
        self.margin2 = margin2

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor, positive, negative1, negative2):
        dist_pos = self.calc_euclidean(anchor, positive)
        dist_neg = self.calc_euclidean(anchor, negative1)
        dist_neg_b = self.calc_euclidean(negative1, negative2)

        loss = (
            F.relu(self.margin1 + dist_pos - dist_neg)
            + F.relu(self.margin2 + dist_pos - dist_neg_b)
        )
        return loss.mean()


class OverallWrapper(nn.Module):
    def __init__(self, lambda_weight=config['lambda_weight']):
        super().__init__()
        self.quadruplet_loss = QuadrupletLoss()
        self.attr_loss = AttributesLoss()
        self.lambda_weight = lambda_weight

    def forward(self, preds, attrs, anchor, positive, first_negative, second_negative):
        return (
            self.lambda_weight * self.attr_loss(preds, attrs)
            + (1 - self.lambda_weight) * self.quadruplet_loss(
                anchor, positive, first_negative, second_negative
            )
        )


def get_cost_function():
    return OverallWrapper()


def get_optimizer(net, lr, wd, momentum):
    return torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=wd)


# ---------------------------------------------------------------------------
# Train / evaluation loops
# ---------------------------------------------------------------------------

def train(net, data_loader, optimizer, cost_function, attribute_names, device='cuda:0'):
    samples = 0.
    cumulative_loss = 0.
    accuracy = [0. for _ in range(len(attribute_names))]

    net.train()
    for batch_idx, (inputs, targets, quadruplet) in tqdm(enumerate(data_loader)):
        inputs = inputs.to(device)
        targets = targets.to(device)
        positive = quadruplet[0].to(device)
        first_negative = quadruplet[1].to(device)
        second_negative = quadruplet[2].to(device)

        outputs = net(inputs)
        anchor = outputs[2]
        positive_out = net(positive)[2]
        first_negative_out = net(first_negative)[2]
        second_negative_out = net(second_negative)[2]

        loss = cost_function(outputs, targets, anchor, positive_out, first_negative_out, second_negative_out)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        predicted = []
        samples += inputs.shape[0]
        cumulative_loss += loss.item()

        for i in range(len(accuracy)):
            if outputs[1][i].size()[1] == 1:
                predicted.append((outputs[1][i] > 0.5).int().flatten())
            else:
                predicted.append(outputs[1][i].max(dim=1)[1])

        for i in range(len(predicted)):
            # FIX: was `predicted[i][1]` — that indexes a single scalar instead of
            # the whole batch tensor, producing wrong accuracy values
            accuracy[i] += predicted[i].eq(targets.transpose(0, 1)[i]).sum().item()

        accuracy_tot = [i / samples * 100 for i in accuracy]
        total_accuracy = sum(accuracy_tot) / len(accuracy_tot)

    return cumulative_loss / samples, [i / samples * 100 for i in accuracy], total_accuracy


def test(net, data_loader, validation_test_loader, validation_queries_loader,
         cost_function, ground_truth_dict, attribute_names, device='cuda:0'):
    samples = 0.
    cumulative_loss = 0.
    accuracy = [0. for _ in range(len(attribute_names))]

    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets, quadruplet) in tqdm(enumerate(data_loader)):
            inputs = inputs.to(device)
            targets = targets.to(device)
            positive = quadruplet[0].to(device)
            first_negative = quadruplet[1].to(device)
            second_negative = quadruplet[2].to(device)

            outputs = net(inputs)
            anchor = outputs[2]
            positive_out = net(positive)[2]
            first_negative_out = net(first_negative)[2]
            second_negative_out = net(second_negative)[2]

            loss = cost_function(outputs, targets, anchor, positive_out, first_negative_out, second_negative_out)

            predicted = []
            samples += inputs.shape[0]
            cumulative_loss += loss.item()

            for i in range(len(accuracy)):
                if outputs[1][i].size()[1] == 1:
                    predicted.append((outputs[1][i] > 0.5).int().flatten())
                else:
                    predicted.append(outputs[1][i].max(dim=1)[1])

            for i in range(len(predicted)):
                accuracy[i] += predicted[i].eq(targets.transpose(0, 1)[i]).sum().item()

            accuracy_tot = [i / samples * 100 for i in accuracy]
            total_accuracy = sum(accuracy_tot) / len(accuracy_tot)

    m_AP = test_mAP(net, validation_test_loader, validation_queries_loader, ground_truth_dict)
    return cumulative_loss / samples, [i / samples * 100 for i in accuracy], total_accuracy, m_AP


# ---------------------------------------------------------------------------
# mAP evaluation
# ---------------------------------------------------------------------------

def get_ground_truth(val_df, val_queries_df):
    values = []
    for idx_q, q in val_queries_df.iterrows():
        matches = {idx_t for idx_t, t in val_df.iterrows() if t['ID'] == q['ID']}
        values.append(matches)
    return dict(zip(range(len(val_queries_df)), values))


def image_feature(net, dataloader, device='cuda:0'):
    images_features = []
    for batch_idx, inputs in enumerate(dataloader):
        inputs = inputs.to(device)
        _, _, features = net(inputs)
        images_features.append(features)
    return torch.cat(images_features, dim=0)


def get_topk_images(model, test_loader, queries_loader, mAP_rank=config['mAP_rank']):
    with torch.no_grad():
        test_features = image_feature(model, test_loader)
        query_features = image_feature(model, queries_loader)

        sims_matrix = torch.empty(query_features.size(0), test_features.size(0))
        for idx, q in enumerate(query_features):
            sims_matrix[idx] = F.cosine_similarity(q, test_features, dim=-1)

        sorted_index = torch.argsort(sims_matrix, dim=1, descending=True)
        top_k = sorted_index.narrow_copy(dim=1, start=0, length=mAP_rank)
    return top_k


def test_mAP(model, test_loader, queries_loader, ground_truth_dict):
    top_k = get_topk_images(model, test_loader, queries_loader)
    predictions_dict = {idx: r for idx, r in enumerate(top_k.tolist())}
    return Evaluator.evaluate_map(predictions_dict, ground_truth_dict)


class Evaluator:
    @staticmethod
    def evaluate_map(predictions: Dict[str, List], ground_truth: Dict[str, Set]) -> float:
        m_ap = 0.0
        for current_gt_query, current_gt_set in ground_truth.items():
            if current_gt_query not in predictions:
                continue
            current_ap = 0.0
            delta_recall = 1.0 / len(current_gt_set)
            encountered_positives = 0
            for idx, pred in enumerate(predictions[current_gt_query]):
                if pred in current_gt_set:
                    encountered_positives += 1
                    current_ap += (encountered_positives / (idx + 1)) * delta_recall
            m_ap += current_ap
        m_ap /= len(ground_truth)
        return m_ap


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def attribute_prediction(model, loader, device='cuda:0'):
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for batch_idx, images in enumerate(loader):
            images = images.to(device)
            _, attributes, _ = model(images)
            predictions = []
            for output in attributes:
                if output.size()[1] == 1:
                    pred = torch.round(torch.squeeze(output, 1))
                else:
                    pred = torch.argmax(output, dim=1)
                predictions.append(torch.unsqueeze(pred, 1))
            all_predictions.append(torch.cat(predictions, dim=1))
    return torch.cat(all_predictions, dim=0).tolist()


def log_values(writer, step, loss, accuracy, total_accuracy, prefix, att):
    writer.add_scalar(f"{prefix}/loss", loss, step)
    writer.add_scalar(f"{prefix}/Total accuracy", total_accuracy)
    for i, name in enumerate(att):
        writer.add_scalar(f"{prefix}/{name} accuracy", accuracy[i], step)


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def main(config=config, device='cuda:0'):
    from torch.utils.tensorboard import SummaryWriter

    complete_df, att_map = build_dataframe(config)
    train_df, valid_df, validation_test_df, validation_queries_df = build_splits(complete_df, config)
    attribute_names = complete_df.columns[3:].tolist()

    datasets = build_datasets(train_df, valid_df, validation_test_df, validation_queries_df, config)
    train_ds, valid_ds, validation_test_ds, validation_queries_ds, test_ds, queries_ds = datasets

    loaders = build_dataloaders(*datasets, config=config)
    (train_loader, val_loader, validation_test_loader,
     validation_queries_loader, test_loader, queries_loader) = loaders

    ground_truth_dict = get_ground_truth(validation_test_df, validation_queries_df)

    writer = SummaryWriter(log_dir='runs/exp1')

    net = Backbone(attribute_map=att_map).to(device)
    optimizer = get_optimizer(net, config['learning_rate'], config['weight_decay'], config['momentum'])
    cost_function = get_cost_function()

    print('Before training:')
    train_loss, train_acc, train_total_acc, _ = test(
        net, train_loader, validation_test_loader, validation_queries_loader,
        cost_function, ground_truth_dict, attribute_names, device,
    )
    val_loss, val_acc, val_total_acc, mAP = test(
        net, val_loader, validation_test_loader, validation_queries_loader,
        cost_function, ground_truth_dict, attribute_names, device,
    )
    log_values(writer, -1, train_loss, train_acc, train_total_acc, 'Train', attribute_names)
    log_values(writer, -1, val_loss, val_acc, val_total_acc, 'Validation', attribute_names)
    writer.add_scalar('mAP_before_training', mAP, 0)

    for e in range(config['epochs']):
        print(f'Epoch {e}')
        train_loss, train_acc, train_total_acc = train(
            net, train_loader, optimizer, cost_function, attribute_names, device
        )
        val_loss, val_acc, val_total_acc, mAP = test(
            net, val_loader, validation_test_loader, validation_queries_loader,
            cost_function, ground_truth_dict, attribute_names, device,
        )
        log_values(writer, e, train_loss, train_acc, train_total_acc, 'Train', attribute_names)
        log_values(writer, e, val_loss, val_acc, val_total_acc, 'Validation', attribute_names)
        writer.add_scalar('mAP', mAP, e)

        print(f'  Train loss: {train_loss:.5f}  Val total acc: {val_total_acc:.2f}  mAP: {mAP:.5f}')

    # FIX: was creating a new untrained Backbone and saving that instead of the
    # trained network — all learned weights would have been lost
    torch.save(net.state_dict(), 'model_trained.pt')
    print('Model saved to model_trained.pt')

    writer.close()

    # --- Task 1: attribute recognition ---
    attributes = attribute_prediction(net, test_loader, device)
    attr_df = pd.DataFrame(
        data=attributes,
        index=sorted(os.listdir(config['test_path'])),
        columns=attribute_names,
    )
    attr_df.to_csv('classification_test.csv', index=True)
    print('Saved classification_test.csv')

    # --- Task 2: person re-identification ---
    final_top_k = get_topk_images(net, test_loader, queries_loader)
    query_list_dir = sorted(os.listdir(config['queries_path']))
    test_list_dir = sorted(os.listdir(config['test_path']))
    lines = [
        query_list_dir[idx] + ': ' + ', '.join(test_list_dir[x] for x in ids) + '\n'
        for idx, ids in enumerate(final_top_k.tolist())
    ]
    with open('reid_test.txt', 'w') as f:
        f.writelines(lines)
    print('Saved reid_test.txt')


if __name__ == '__main__':
    main()
