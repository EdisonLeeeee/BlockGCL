import copy
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F

from absl import app, flags
from torch.optim import AdamW

# custom modules
from logger import setup_logger
from utils import set_random_seeds, edgeidx2sparse
from transforms import get_graph_drop_transform
from model import GCN
from loss import inv_dec_loss
from eval import test, batch_test
from dataloader import load_data

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    'model_seed', 123, 'Random seed used for model initialization and training.')
flags.DEFINE_integer(
    'data_seed', 0, 'Random seed used to generate train/val/test split.')
flags.DEFINE_integer('gpu_id', 0, 'The id of GPU to use. -1 indicates CPU.')
# Dataset.
flags.DEFINE_enum('dataset', 'ogbn-mag',
                  ['Cora', 'Citeseer', 'Pubmed', 'Computers', 'Photo',
                   'CS', 'Physics', 'WikiCS', 'ogbn-arxiv', 'ogbn-mag'],
                  'Which graph dataset to use.')
flags.DEFINE_string('data_dir', '~/public_data/pyg_data/',
                    'Where the dataset resides.')

# Architecture.
flags.DEFINE_multi_integer('graph_encoder_layer', [
                           256, 256], 'Conv layer sizes.')
flags.DEFINE_bool('batchnorm', True, 'Batchnorm or not.')
flags.DEFINE_string('layer_name', "GCN", 'Con. layer.')
flags.DEFINE_string('act_name', "ReLU", 'Activation funciton.')

# Training hyperparameters.
flags.DEFINE_float('lambd', 1e-3, 'The ratio for decorrelation loss.')
flags.DEFINE_integer('epochs', 500, 'The number of training epochs.')
flags.DEFINE_float('lr', 1e-3, 'The learning rate for model training.')
flags.DEFINE_float('weight_decay', 1e-5,
                   'The value of the weight decay for training.')
flags.DEFINE_float(
    'lr_cls', 1e-2,
    'The learning rate for model training for downstream classifier.')
flags.DEFINE_float(
    'wd_cls', 1e-5,
    'The value of the weight decay for training for downstream classifier..')
flags.DEFINE_integer(
    'epochs_cls', 100,
    'The number of training epochs for node downstream classifier.')

# Augmentations.
flags.DEFINE_float('drop_edge_p', 0.4, 'Probability of edge dropout 1.')
flags.DEFINE_float('drop_feat_p', 0.2,
                   'Probability of node feature dropout 1.')

# Logging and checkpoint.
flags.DEFINE_string(
    'logdir', None, 'Where the checkpoint and logs are stored.')
flags.DEFINE_string('mask_dir', './mask',
                    'Where the checkpoint and logs are stored.')

# Evaluation
flags.DEFINE_integer('eval_period', 5, 'Evaluate every eval_epochs.')


def run(dataset, logger):
    gpu_available = torch.cuda.is_available() and FLAGS.gpu_id >= 0
    device = torch.device("cuda:{}".format(FLAGS.gpu_id)) if gpu_available \
        else torch.device("cpu")
    logger.info("Using {} for training.".format(device))

    data = dataset[0].to(device)
    num_classes = dataset.num_classes

    # set random seed
    if FLAGS.model_seed is not None:
        set_random_seeds(FLAGS.model_seed)
        logger.info("Random seed set to {}.".format(FLAGS.model_seed))

    transform = get_graph_drop_transform(drop_edge_p=FLAGS.drop_edge_p,
                                         drop_feat_p=FLAGS.drop_feat_p)
    
    encoder = GCN(data.x.size(1), FLAGS.graph_encoder_layer, FLAGS).to(device)
    optimizer = AdamW(params=[{"params" :encoder.parameters()}], 
                      lr=FLAGS.lr,
                      weight_decay=FLAGS.weight_decay)

    # number of parameters
    total_params = sum([param.nelement() for param in encoder.parameters()])
    logger.info(encoder)
    logger.info("Number of parameter: %.2fM" % (total_params/1e6))

    # start training
    logger.info("Satrt training")

    best_test_acc_mean, best_test_acc_std, \
        best_test_acc_epoch, best_test_acc_list = 0, 0, 0, []

    for epoch in range(1, 1 + FLAGS.epochs):
        # torch.cuda.empty_cache()
        encoder.train()
        optimizer.zero_grad()

        data1 = transform(data)
        data2 = transform(data)
        data1.edge_index = edgeidx2sparse(data1.edge_index, data1.x.size(0))
        data2.edge_index = edgeidx2sparse(data2.edge_index, data2.x.size(0))
        
        outputs1, outputs2 = encoder(data1.x, data1.edge_index), encoder(data2.x, data2.edge_index)
        
        total_loss = 0.
        for o1, o2 in list(zip(outputs1, outputs2)):
            loss = inv_dec_loss(o1, o2, FLAGS.lambd)
            # total_loss += loss
            total_loss += loss.item()
            loss.backward()

        # total_loss.backward()
        optimizer.step()

        # eval
        if epoch == 1 or epoch % FLAGS.eval_period == 0:
            encoder.eval()
            with torch.no_grad():
                # embeds = torch.cat(encoder(data), dim=1)
                embeds = encoder(data.x, data.edge_index)[-1]
                # embeds = encoder.embeds(data.x, data.edge_index)
                # embeds = torch.cat(embeds, dim=1)

            if FLAGS.dataset in ['ogbn-arxiv', 'ogbn-mag']:
                _, test_acc_list = batch_test(embeds=embeds,
                                              data=data,
                                              num_classes=num_classes,
                                              FLAGS=FLAGS,
                                              device=device)
            else:
                _, test_acc_list = test(embeds=embeds,
                                        data=data,
                                        num_classes=num_classes,
                                        FLAGS=FLAGS,
                                        device=device)

            test_acc_mean, test_acc_std = \
                np.mean(test_acc_list), np.std(test_acc_list)

            if test_acc_mean > best_test_acc_mean:
                best_test_acc_mean = test_acc_mean
                best_test_acc_std = test_acc_std
                best_test_acc_epoch = epoch
                best_test_acc_list = copy.deepcopy(test_acc_list)
                # save encoder weights
                # torch.save(model.online_encoder.state_dict(), os.path.join(FLAGS.logdir, '{}.pt'.format(FLAGS.dataset)))

            logger.info("[Epoch {:4d}/{:4d}] loss={:.4f}, "
                        "test_acc={:.2f}±{:.2f} "
                        "[best_test_acc: {:.2f}±{:.2f} at epoch {}]".format(
                            epoch, FLAGS.epochs, total_loss,
                            test_acc_mean * 100, test_acc_std * 100,
                            best_test_acc_mean * 100, best_test_acc_std * 100,
                            best_test_acc_epoch
                        ))

    logger.info("Best test acc: {:.2f}±{:.2f} at epoch {}: {}".format(
        best_test_acc_mean * 100, best_test_acc_std * 100,
        best_test_acc_epoch, best_test_acc_list
    ))


def get_dataset():
    dataset = load_data(data_dir=osp.expanduser(FLAGS.data_dir),
                        dataset_name=FLAGS.dataset,
                        mask_dir=FLAGS.mask_dir,
                        load_mask=False,
                        save_mask=False)
    return dataset


def main(argv):
    logger = setup_logger(output="./logs/exp.log".format(FLAGS.dataset))
    dataset = get_dataset()
    run(dataset=dataset, logger=logger)


if __name__ == "__main__":
    app.run(main)
