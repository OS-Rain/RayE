import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score

from models import AttentionAgger
from utils import Writer, Criterion, save_checkpoint, get_preds, get_lr, set_seed, process_data, get_energy_softmax
from utils import get_local_config_name, get_model, get_data_loaders, write_stat_from_metric_dicts, init_metric_dict

from ogb.graphproppred import Evaluator

class RayE(nn.Module):
    def __init__(self, clf, attenaggr, optimizer, scheduler, writer, device, model_dir, dataset_name, num_class, multi_label, random_state,
                 method_config, shared_config):
        super().__init__()
        self.clf = clf
        self.attenaggr = attenaggr
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.writer = writer
        self.device = device
        self.model_dir = model_dir
        self.dataset_name = dataset_name
        self.random_state = random_state
        self.method_name = method_config['method_name']
        self.beta = method_config['beta']
        self.learn_edge_att = shared_config['learn_edge_att']
        self.epochs = method_config['epochs']
        self.multi_label = multi_label
        self.criterion = Criterion(num_class, multi_label)


    def __loss__(self, clf_logits, boundary_x, clf_labels, epoch):
        loss = self.criterion(clf_logits, clf_labels)
        loss_dict = {'loss': loss.item()}
        return loss, loss_dict

    def forward_pass(self, data, epoch, training):
        emb_nodes_l1 = self.clf.get_nodes_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
        emb_global_l1 = self.clf.get_global_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
        
        edge_index, boundary_x = get_energy_softmax(emb_nodes_l1, data.edge_index, data.resistance, data.batch, epoch, beta=self.beta)
        emb_nodes_l2 = self.clf.get_nodes_emb(data.x, edge_index, batch=data.batch, edge_attr=data.edge_attr)
        clf_logits = self.attenaggr.get_emb(Q=emb_global_l1, K=emb_nodes_l2, V=emb_nodes_l2)

        loss, loss_dict = self.__loss__(clf_logits, boundary_x,  data.y, epoch)

        return loss, loss_dict, clf_logits

    @torch.no_grad()
    def eval_one_batch(self, data, epoch):
        self.clf.eval()
        self.attenaggr.eval()

        loss, loss_dict, clf_logits = self.forward_pass(data, epoch, training=False)
        return loss_dict, clf_logits.data.cpu()

    def train_one_batch(self, data, epoch):
        self.clf.train()
        self.attenaggr.train()

        loss, loss_dict, clf_logits = self.forward_pass(data, epoch, training=True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss_dict, clf_logits.data.cpu()

    def run_one_epoch(self, data_loader, epoch, phase, use_edge_attr):
        loader_len = len(data_loader)
        run_one_batch = self.train_one_batch if phase == 'train' else self.eval_one_batch
        phase = 'test ' if phase == 'test' else phase  # align tqdm desc bar

        all_loss_dict = {}
        all_exp_labels, all_clf_labels, all_clf_logits = ([] for i in range(3))
        pbar = tqdm(data_loader)
        for idx, data in enumerate(pbar):
            data = process_data(data, use_edge_attr)
            loss_dict, clf_logits = run_one_batch(data.to(self.device), epoch)

            exp_labels = data.edge_label.data.cpu()
            desc, _, _, _, = self.log_epoch(epoch, phase, loss_dict, exp_labels,
                                                 data.y.data.cpu(), clf_logits, batch=True)
            for k, v in loss_dict.items():
                all_loss_dict[k] = all_loss_dict.get(k, 0) + v

            all_exp_labels.append(exp_labels)
            all_clf_labels.append(data.y.data.cpu()), all_clf_logits.append(clf_logits)

            if idx == loader_len - 1:
                all_exp_labels = torch.cat(all_exp_labels)
                all_clf_labels, all_clf_logits = torch.cat(all_clf_labels), torch.cat(all_clf_logits)

                for k, v in all_loss_dict.items():
                    all_loss_dict[k] = v / loader_len
                desc, clf_acc, clf_roc, avg_loss = self.log_epoch(epoch, phase, all_loss_dict, all_exp_labels,
                                                                                         all_clf_labels, all_clf_logits, batch=False)
            pbar.set_description(desc)
        return  clf_acc, clf_roc, avg_loss

    def train(self, loaders, test_set, metric_dict, use_edge_attr):
        for epoch in range(self.epochs):
            train_res = self.run_one_epoch(loaders['train'], epoch, 'train', use_edge_attr)
            valid_res = self.run_one_epoch(loaders['valid'], epoch, 'valid', use_edge_attr)
            test_res = self.run_one_epoch(loaders['test'], epoch, 'test', use_edge_attr)
            self.writer.add_scalar('RayE_train/lr', get_lr(self.optimizer), epoch)

            assert len(train_res) == 3
            main_metric_idx = 1 if 'ogb' in self.dataset_name  else 0  # clf_roc or clf_acc
            
            if self.scheduler is not None:
                self.scheduler.step(valid_res[main_metric_idx])

            if  epoch > 10 and ((valid_res[main_metric_idx] > metric_dict['metric/best_clf_valid'])
                                                                     or (valid_res[main_metric_idx] == metric_dict['metric/best_clf_valid']
                                                                         and valid_res[2] < metric_dict['metric/best_clf_valid_loss'])):

                metric_dict = {'metric/best_clf_epoch': epoch, 'metric/best_clf_valid_loss': valid_res[2],
                               'metric/best_clf_train': train_res[main_metric_idx], 'metric/best_clf_valid': valid_res[main_metric_idx],'metric/best_clf_test': test_res[main_metric_idx],}
                save_checkpoint(self.clf, self.model_dir, model_name='RayE_clf_epoch_' + str(epoch))
                save_checkpoint(self.attenaggr, self.model_dir, model_name='RayE_attenaggr_epoch_' + str(epoch))

            for metric, value in metric_dict.items():
                metric = metric.split('/')[-1]
                self.writer.add_scalar(f'RayE_best/{metric}', value, epoch)

            if epoch == self.epochs - 1:
                save_checkpoint(self.clf, self.model_dir, model_name='RayE_clf_epoch_' + str(epoch))
                save_checkpoint(self.attenaggr, self.model_dir, model_name='RayE_attenaggr_epoch_' + str(epoch))

            print(f'[Seed {self.random_state}, Epoch: {epoch}]: Best Epoch: {metric_dict["metric/best_clf_epoch"]}, '
                  f'Best Val Pred ACC/ROC: {metric_dict["metric/best_clf_valid"]:.3f}, Best Test Pred ACC/ROC: {metric_dict["metric/best_clf_test"]:.3f}')
            print('====================================')
            print('====================================')
        return metric_dict


    def log_epoch(self, epoch, phase, loss_dict, exp_labels, clf_labels, clf_logits, batch):
        desc = f'[Seed {self.random_state}, Epoch: {epoch}]: RayE_{phase}........., ' if batch else f'[Seed {self.random_state}, Epoch: {epoch}]: RayE_{phase} finished, '
        for k, v in loss_dict.items():
            if not batch:
                self.writer.add_scalar(f'RayE_{phase}/{k}', v, epoch)
            desc += f'{k}: {v:.3f}, '

        eval_desc, clf_acc, clf_roc = self.get_eval_score(epoch, phase, exp_labels, clf_labels, clf_logits, batch)
        desc += eval_desc
        return desc, clf_acc, clf_roc, loss_dict['loss']



    def get_eval_score(self, epoch, phase, exp_labels, clf_labels, clf_logits, batch):
        clf_preds = get_preds(clf_logits, self.multi_label)
        clf_acc = 0 if self.multi_label else (clf_preds.reshape([clf_preds.shape[0]]) == clf_labels.reshape([clf_labels.shape[0]])).sum().item() / clf_labels.shape[0]
        if batch:
            return f'clf_acc: {clf_acc:.3f}', None, None

        clf_roc = 0
        if 'ogb' in self.dataset_name :
            evaluator = Evaluator(name='-'.join(self.dataset_name.split('_')))
            clf_roc = evaluator.eval({'y_pred': clf_logits, 'y_true': clf_labels})['rocauc']

        self.writer.add_scalar(f'RayE_{phase}/clf_acc/', clf_acc, epoch)
        self.writer.add_scalar(f'RayE_{phase}/clf_roc/', clf_roc, epoch)


        desc = f'clf_acc: {clf_acc:.3f}, clf_roc: {clf_roc:.3f}, ' 
        return desc, clf_acc, clf_roc


def train_RayE_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, method_name, device, random_state):
    print('====================================')
    print('====================================')
    print(f'[INFO] Using device: {device}')
    print(f'[INFO] Using random_state: {random_state}')
    print(f'[INFO] Using dataset: {dataset_name}')
    print(f'[INFO] Using model: {model_name}')

    set_seed(random_state)

    model_config = local_config['model_config']
    data_config = local_config['data_config']
    method_config = local_config[f'{method_name}_config']
    shared_config = local_config['shared_config']
    assert model_config['model_name'] == model_name
    assert method_config['method_name'] == method_name

    batch_size, splits = data_config['batch_size'], data_config.get('splits', None)
    loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info = get_data_loaders(data_dir, dataset_name, batch_size, splits, random_state, data_config.get('mutag_x', False))

    model_config['deg'] = aux_info['deg']
    model = get_model(x_dim, edge_attr_dim, num_class, aux_info['multi_label'], model_config, device)

    print('====================================')
    print('====================================')

    log_dir.mkdir(parents=True, exist_ok=True)

    # if not method_config['from_scratch']:
    #     print('[INFO] Pretraining the model...')
    #     train_clf_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, device, random_state,
    #                        model=model, loaders=loaders, num_class=num_class, aux_info=aux_info)
    #     pretrain_epochs = local_config['model_config']['pretrain_epochs'] - 1
    #     load_checkpoint(model, model_dir=log_dir, model_name=f'epoch_{pretrain_epochs}')
    # else:
    #     print('[INFO] Training both the model and the attention from scratch...')

    attenaggr = AttentionAgger(model_config['hidden_size'], model_config['hidden_size'], model_config['hidden_size'], num_class, aux_info['multi_label']).to(device)

    lr, wd = method_config['lr'], method_config.get('weight_decay', 0)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(attenaggr.parameters()), lr=lr, weight_decay=wd)

    scheduler_config = method_config.get('scheduler', {})
    scheduler = None if scheduler_config == {} else ReduceLROnPlateau(optimizer, mode='max', **scheduler_config)

    writer = Writer(log_dir=log_dir)
    hparam_dict = {**model_config, **data_config}
    hparam_dict = {k: str(v) if isinstance(v, (dict, list)) else v for k, v in hparam_dict.items()}
    metric_dict = deepcopy(init_metric_dict)
    writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)

    print('====================================')
    print('[INFO] Training RayE...')
    rayE = RayE(model, attenaggr, optimizer, scheduler, writer, device, log_dir, dataset_name, num_class, aux_info['multi_label'], random_state, method_config, shared_config)
    metric_dict = rayE.train(loaders, test_set, metric_dict, model_config.get('use_edge_attr', True))
    writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)
    return hparam_dict, metric_dict


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train RayE')
    parser.add_argument('--dataset', type=str, help='dataset used')
    parser.add_argument('--backbone', type=str, help='backbone model used')
    parser.add_argument('--cuda', type=int, help='cuda device id, -1 for cpu')
    args = parser.parse_args()
    dataset_name = args.dataset
    model_name = args.backbone
    cuda_id = args.cuda

    torch.set_num_threads(5)
    config_dir = Path('./configs')
    method_name = 'RayE'

    print('====================================')
    print('====================================')
    print(f'[INFO] Running {method_name} on {dataset_name} with {model_name}')
    print('====================================')

    global_config = yaml.safe_load((config_dir / 'global_config.yml').open('r'))
    local_config_name = get_local_config_name(model_name, dataset_name)
    local_config = yaml.safe_load((config_dir / local_config_name).open('r'))

    data_dir = Path(global_config['data_dir'])
    num_seeds = global_config['num_seeds']

    time = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
    device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu')

    metric_dicts = []
    for random_state in range(num_seeds):
        log_dir = data_dir / dataset_name / 'logs' / (time + '-' + dataset_name + '-' + model_name + '-seed' + str(random_state) + '-' + method_name)
        hparam_dict, metric_dict = train_RayE_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, method_name, device, random_state)
        metric_dicts.append(metric_dict)

    log_dir = data_dir / dataset_name / 'logs' / (time + '-' + dataset_name + '-' + model_name + '-seed99-' + method_name + '-stat')
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = Writer(log_dir=log_dir)
    write_stat_from_metric_dicts(hparam_dict, metric_dicts, writer)


if __name__ == '__main__':
    main()
