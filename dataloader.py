import os.path as osp
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WikiCS
from torch_geometric.transforms import Compose, NormalizeFeatures, ToUndirected
from ogb.nodeproppred import PygNodePropPredDataset


def load_data(data_dir, dataset_name,
              transform=Compose([ToUndirected()]),
              mask_dir="./mask",
              load_mask=True,
              save_mask=True):
    """Load PyG dataset."""
    load_mask = save_mask = True

    if dataset_name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(root=data_dir, name=dataset_name,
                            transform=transform, split="full")
        load_mask = save_mask = False

    elif dataset_name in ['WikiCS']:
        dataset = WikiCS(root=osp.join(data_dir, dataset_name),
                         transform=transform)

    elif dataset_name in ['Computers', 'Photo']:
        dataset = Amazon(root=data_dir, name=dataset_name, transform=transform)

    elif dataset_name in ['CS', 'Physics']:
        dataset = Coauthor(root=data_dir, name=dataset_name,
                           transform=transform)

    elif dataset_name in ['ogbn-arxiv']:
        dataset = PygNodePropPredDataset(root=data_dir, name=dataset_name,
                                         transform=transform)
        dataset.data.y = dataset.data.y.squeeze()
        load_mask = save_mask = False
    elif dataset_name in ['ogbn-mag']:
        dataset = PygNodePropPredDataset(name=dataset_name, root=data_dir,
                                transform=Compose([
                                    ToUndirected()
                                ]))
        rel_data = dataset[0]
        # We are only interested in paper <-> paper relations.
        data = Data(
                x=rel_data.x_dict['paper'],
                edge_index=rel_data.edge_index_dict[('paper', 'cites', 'paper')],
                y=rel_data.y_dict['paper'])
        data = transform(data)
        dataset.data = data        
        dataset.data.y = dataset.data.y.squeeze()
        load_mask = save_mask = False

    else:
        raise ValueError("Dataset {} not implemented.".format(dataset_name))

    mask_path = osp.join(mask_dir, "{}.pt".format(dataset_name))

    if osp.exists(mask_path) and load_mask:
        train_mask, val_mask, test_mask = load_preset_mask(mask_path)
    else:
        train_mask, val_mask, test_mask = create_mask(
            dataset=dataset,
            dataset_name=dataset_name,
            mask_path=mask_path if save_mask else None)

    dataset.data.train_mask = train_mask
    dataset.data.val_mask = val_mask
    dataset.data.test_mask = test_mask

    return dataset


def create_mask(dataset, dataset_name='WikiCS', data_seed=0, mask_path=None):
    r"""Create train/val/test mask for each dataset."""
    data = dataset[0]
    if dataset_name in ['Cora', 'Citeseer', 'Pubmed']:
        train_mask, val_mask, test_mask = \
            data.train_mask, data.val_mask, data.test_mask

    elif dataset_name in ['WikiCS']:
        train_mask = data.train_mask.t()
        val_mask = data.val_mask.t()
        test_mask = data.test_mask.repeat(20, 1)

    elif dataset_name in ['Computers', 'Photo', 'CS', 'Physics']:
        idx = np.arange(len(data.y))

        train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
        val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
        test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)

        train_idx, test_idx = train_test_split(
            idx, test_size=0.8, random_state=data_seed)
        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.5, random_state=data_seed)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

    elif dataset_name in ['ogbn-arxiv']:
        split_idx = dataset.get_idx_split()
        train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
        val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
        test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
        train_mask[split_idx['train']] = True
        val_mask[split_idx['valid']] = True
        test_mask[split_idx['test']] = True
    elif dataset_name in ['ogbn-mag']:
        split_idx = dataset.get_idx_split()
        train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
        val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
        test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
        train_mask[split_idx['train']['paper']] = True
        val_mask[split_idx['valid']['paper']] = True
        test_mask[split_idx['test']['paper']] = True          

    # save preset mask
    if mask_path is not None:
        torch.save([train_mask, val_mask, test_mask], mask_path)

    return train_mask, val_mask, test_mask


def load_preset_mask(mask_path):
    return torch.load(mask_path)
