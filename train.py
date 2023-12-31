"""
Jiarui Feng
File for training and evaluation(K-fold)
Modified by Zitian Tang 02/2023
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import utils
from collections import OrderedDict
from json import dumps
from layers import GCNLayer,GATLayer
from models import make_gnn_layer,GNN,GraphRegression
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.optim import Adam
from functools import partial
import torch.utils.data as data
from torch_geometric.data import  DataLoader
from torch_geometric.data import Batch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import ExponentialLR
import argparse
import torch_geometric.transforms as T
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import recall_score, f1_score
import pandas as pd
import os
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def get_model(args, log,load_path=None):
    gnn_layer = make_gnn_layer(gnn_type=args.gnn_type,
                               hidden_size=args.hidden_size,
                               head=args.head,
                               num_edge_type=args.num_edge_type)

    gnn = GNN(input_size=args.input_size,
              hidden_size=args.hidden_size,
              num_layer=args.num_layer,
              gnn_layer=gnn_layer,
              JK=args.JK,
              norm_type=args.norm_type,
              edge_drop_prob=args.edge_drop_prob,
              drop_prob=args.drop_prob)

    model = GraphRegression(embedding_model=gnn,
                            pooling_method=args.pooling_method
                            )

    if load_path is not None:
        log.info(f'Loading checkpoint from {load_path}...')
        model.load_state_dict(torch.load(load_path))
        step = 20000

    else:
        step = 0

    return model, step


def k_fold(num_data, folds,seed):
    kf = KFold(folds, shuffle=True, random_state=seed)

    test_indices, train_indices = [], []
    for _, idx in kf.split(torch.zeros(num_data)):
        test_indices.append(torch.from_numpy(idx).long())

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(num_data).long()
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices


def val1(data_loader,model,device):
    model.eval()
    loss_all = 0
    num_data=len(data_loader.dataset)
    pre=torch.zeros([num_data])
    target=torch.zeros([num_data])
    with torch.no_grad(), \
         tqdm(total=len(data_loader.dataset)) as progress_bar:
        for i,batch_graphs in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_y=batch_graphs.y
            batch_size=batch_graphs.num_graphs
            predict=model(batch_graphs)
            if len(predict) == 2:
                predict, attention = predict
            loss=F.mse_loss(predict,batch_y,reduction='sum')
            loss_all += loss
            progress_bar.update(batch_size)
            pre[batch_size*i:batch_size*(i+1)]=predict
            target[batch_size*i:batch_size*(i+1)]=batch_y


    model.train()
    avg_loss=loss_all / len(data_loader.dataset)
    avg_loss=avg_loss.item()
    score_list=[y.item() for y in target]
    predict_score_list=[predict.item() for predict in pre]
    corr_dict = {'Score': score_list, 'Pred Score': predict_score_list}
    corr_df = pd.DataFrame(corr_dict)
    corr=corr_df.corr(method="pearson")
    corr=corr['Pred Score'][0]
    results_list = [
        ('Loss', avg_loss),
        ('Corr', corr)
    ]
    return  OrderedDict(results_list)


def val(data_loader,model,device,current_name):
    model.eval()
    loss_all = 0
    num_data=len(data_loader.dataset)
    pre=torch.zeros([num_data])
    target=torch.zeros([num_data])
    loss_list = []
    attention_total = []
    with torch.no_grad(), \
         tqdm(total=len(data_loader.dataset)) as progress_bar:
        for i,batch_graphs in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_y=batch_graphs.y
            batch_size=batch_graphs.num_graphs
            predict=model(batch_graphs)
            if len(predict) == 2:
                predict, attention = predict
            attention = attention.cpu().detach().numpy() # original shape: (293584, 2, 4)
            num_samples = attention.shape[0] / 294737
            attention = attention.reshape(int(num_samples), 294737, 2, 4)
            attention_total.append(attention)
            # batch size 16, edge num 18349, xxx 2, hum head 4 (final batch 9)
            if len(predict.size()) == 1:
                predict = predict.unsqueeze(0)

            loss = F.cross_entropy(predict, batch_y)
            loss_batch = []
            for b in range(batch_size):
                loss_batch.append(F.cross_entropy(predict[b], batch_y[b]).cpu().detach().numpy())
            loss_list.append(loss_batch)

            loss_all += loss.item()*batch_size
            progress_bar.update(batch_size)
            pre[batch_size*i:batch_size*(i+1)]=predict.argmax(-1)
            target[batch_size*i:batch_size*(i+1)]=batch_y

    model.train()

    pd.DataFrame(loss_list).to_csv(f'../data/{current_name}_loss_list.csv', index=False)
    attention_total = np.concatenate(attention_total, axis=0)
    np.savez(f'./save/attention/attention_{current_name}.npz', attention=attention_total)
    acc = (pre == target).sum() / num_data
    # add recall and f1 score
    recall = recall_score(target.cpu().numpy(), pre.cpu().numpy(), average='weighted')
    f1 = f1_score(target.cpu().numpy(), pre.cpu().numpy(), average='weighted')
    loss = loss_all / num_data
    # result = {"Accuracy": acc, "Loss": loss}
    result = {"Accuracy": acc, "Loss": loss, "Recall": recall, "F1 Score": f1}
    return result
    # return  OrderedDict(results_list)


def main():
    parser = argparse.ArgumentParser(f'arguments for training and evaluation')
    parser.add_argument('--name',type=str,default="Gr4_allsamples",help='Name of the experiment.')
    parser.add_argument('--save_dir',type=str,default='./save/',help='Base directory for saving information.')
    parser.add_argument('--data_path',type=str,default='../data/processed_data.npz',help='data location.')
    parser.add_argument('--seed',type=int,default=224,help='Random seed for reproducibility.')
    parser.add_argument('--drop_prob',type=float,default=0.1,help='Probability of zeroing an activation in dropout layers.')
    parser.add_argument('--edge_drop_prob',type=float,default=0.,help='Probability of dropping edge in the gnn training.')
    parser.add_argument('--batch_size',type=int,default=4,help='Batch size per GPU. Scales automatically when \
                            multiple GPUs are available.')
    parser.add_argument('--num_workers',type=int,default=0,help='number of worker.')
    parser.add_argument('--load_path',type=str,default='./save/train/MB_project',help='Path to load as a model checkpoint.')
    parser.add_argument('--gnn_type',type=str,default="GNN",choices=("GAT","GCN","GNN"),help='Type of gnn layer')
    parser.add_argument('--lr',type=float,default=0.0005,help='Learning rate.')
    parser.add_argument('--min_lr',type=float,default=1e-7,help='minimum Learning rate.')
    parser.add_argument('--l2_wd',type=float,default=0.0,help='L2 weight decay.')
    parser.add_argument('--ema_decay',type=float,default=0.999,help='Decay rate for exponential moving average of parameters.')
    parser.add_argument('--num_epochs',type=int,default=25,help='Number of epochs.') # prev 40
    parser.add_argument('--metric_name',type=str,default="Accuracy",choices=("Loss","Accuracy","F1","AUC","Corr"),help='Name of dev metric to determine best checkpoint.')
    parser.add_argument('--max_checkpoints',type=int,default=5,help='Maximum number of checkpoints to keep on disk.')
    parser.add_argument('--max_grad_norm',type=float,default=5.0,help='Maximum gradient norm for gradient clipping.')
    parser.add_argument("--hidden_size",type=int,default=128,help="hidden size of the model")  # prev 160
    parser.add_argument("--input_size",type=int,default=2,help="input size of the model")
    parser.add_argument("--num_edge_type",type=int,default=0,help="number of edge type")
    parser.add_argument("--num_layer",type=int,default=2,help="number of layer for feature encoder")  # prev 2
    parser.add_argument("--head",type=int,default=4,help="number of head")
    parser.add_argument("--JK",type=str,default="attention",choices=("sum","max","mean","attention","last"),help="Jumping knowledge method")
    parser.add_argument("--pooling_method",type=str,default="mean",choices=("mean","sum","attention"),help="pooling method in graph classification")
    parser.add_argument('--norm_type',type=str,default="Layer",choices=("Batch","Layer","Instance","GraphSize","Pair"),
                        help="normalization method in model")
    parser.add_argument('--aggr',type=str,default="add",help='aggregation method in GNN layer, only works in GraphSAGE')
    parser.add_argument('--split',type=int,default=5,help='number of fold in cross validation')
    parser.add_argument('--eval_steps',type=int,default=500,help="step between the evaluation")

    args=parser.parse_args()
    if args.metric_name in ('Loss'):
        # Best checkpoint is the one that minimizes loss
        args.maximize_metric = False
        args.mode="min"
    elif args.metric_name in ("Accuracy","F1","AUC","Corr"):
        # Best checkpoint is the one that maximizes Accuracy or recall
        args.maximize_metric = True
        args.mode="max"
    else:
        raise ValueError(f'Unrecognized metric name: "{args.metric_name}"')



    # Set up logging and devices
    args.save_dir = utils.get_save_dir(args.save_dir, args.name, type="train")
    log = utils.get_logger(args.save_dir, args.name)
    device, args.gpu_ids = utils.get_available_devices()
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))


    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    input_data = np.load(args.data_path, allow_pickle=True)
    edge_index = input_data["edge_index"]
    expression = input_data["input"]
    y = input_data["output"]
    mask = input_data["mask"]
    ids = input_data["ids"]

    expression[..., 0] = expression[..., 0] * 100
    expression[..., 1] = expression[..., 1] * 10

    # shuffle data before k-fold cross validation
    p = np.random.permutation(expression.shape[0])
    shuffled_expression = expression[p]
    shuffled_y = y[p]

    # split for k-fold cross validation, use one fold for validation, one fold for test, other for training
    train_indices, test_indices, val_indices=k_fold(shuffled_expression.shape[0],args.split,args.seed)

    best_val_loss=np.zeros(args.split)
    best_val_corr=np.zeros(args.split)
    best_test_loss=np.zeros(args.split)
    best_test_corr=np.zeros(args.split)
    # add recall and f1 score
    best_val_recall=np.zeros(args.split)
    best_test_recall=np.zeros(args.split)
    best_val_f1=np.zeros(args.split)
    best_test_f1=np.zeros(args.split)
    ##
    for k in range(args.split): # args.split (or 1 for 1 fold)
        log.info(f"---------------Training and evaulation on fold {k+1}------------------------")
        model,step=get_model(args,log)
        model.to(device)
        model.train()
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        log.info(f'The total parameters of model :{[pytorch_total_params]}')

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.l2_wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=args.mode, factor=0.5, patience=3, min_lr=args.min_lr)
        #
        # # normalization according to train mean & std
        # train_mean = np.mean(shuffled_expression[train_indices], axis=0)
        # train_std = np.std(shuffled_expression[train_indices], axis=0)
        # shuffled_expression[train_indices] = (shuffled_expression[train_indices] - train_mean) / train_std
        # shuffled_expression[val_indices] = (shuffled_expression[val_indices] - train_mean) / train_std
        # shuffled_expression[test_indices] = (shuffled_expression[test_indices] - train_mean) / train_std

        train_dataset=utils.LoadDataset(shuffled_expression, edge_index, shuffled_y, train_indices[k], mask, ids)
        val_dataset=utils.LoadDataset(shuffled_expression, edge_index, shuffled_y, val_indices[k], mask, ids)
        test_dataset=utils.LoadDataset(shuffled_expression, edge_index, shuffled_y, test_indices[k], mask, ids)

        train_loader=DataLoader(train_dataset,
                                     batch_size=args.batch_size,
                                     collate_fn=utils.collate_fn,
                                     num_workers=args.num_workers,
                                     shuffle=True)

        val_loader=DataLoader(val_dataset,
                                     batch_size=args.batch_size,
                                     collate_fn=utils.collate_fn,
                                     num_workers=args.num_workers,
                                     shuffle=False)

        test_loader=data.DataLoader(test_dataset,
                                     batch_size=args.batch_size,
                                     collate_fn=utils.collate_fn,
                                     num_workers=args.num_workers,
                                     shuffle=False)
        if args.maximize_metric:
            best_metric=0
        else:
            best_metric=1e30
        steps_till_eval = args.eval_steps
        step=0
        trl = tracc = vl = vacc = tl = tsacc = []
        for epoch in range(args.num_epochs):
            log.info(f'Starting epoch {epoch+1}...')
            if optimizer.param_groups[0]['lr'] <= args.min_lr:
                print('lr eq to min lr')
                break

            with torch.enable_grad(), tqdm(total=len(train_loader.dataset)) as progress_bar:
                print("trainloader",len(train_loader.dataset))
                for graphs in train_loader:
                    optimizer.zero_grad()
                    graphs=graphs.to(device)
                    y=graphs.y
                    batch_size=graphs.num_graphs
                    optimizer.zero_grad()
                    predict=model(graphs)

                    if len(predict)==2:
                        predict,attention= predict
                    # x_np = attention.cpu().detach().numpy()
                    # x_df = pd.DataFrame(x_np[:,:,0].reshape(318240,-1))
                    # x_df.to_csv('attention.csv')
                    if len(predict.size()) == 1:
                        predict = predict.unsqueeze(0)
                    loss=F.cross_entropy(predict, y)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    model.zero_grad()
                    # Log info
                    step += batch_size
                    progress_bar.update(batch_size)
                    loss_val = loss.item()
                    lr = optimizer.param_groups[0]['lr']

                    progress_bar.set_postfix(epoch=epoch+1,
                                             loss=loss_val,
                                               lr=lr)
                    steps_till_eval -= batch_size
                    if steps_till_eval <= 0:
                        steps_till_eval = args.eval_steps
                        log.info(f"evaluate at step {step}...")
                        train_result = val(train_loader, model, device, args.name)
                        val_result = val(val_loader,model,device, args.name)
                        test_result =val(test_loader,model,device, args.name)

                        val_metric=val_result[args.metric_name]
                        if args.maximize_metric:
                            if val_metric>=best_metric:
                                best_metric=val_metric
                                log.info(f"Best val metric updated: {best_metric}")
                                log.info(f"Best model saved")
                                torch.save(model.state_dict(), args.save_dir + f'/best_train_model_fold_{k}.pth')
                                # torch.save(attention,  args.save_dir + f'/best_attention_fold_{k}.pth')
                        else:
                            if val_metric<=best_metric:
                                best_metric=val_metric
                                log.info(f"Best val metric updated: {best_metric}")
                                log.info(f"Best model saved")
                                torch.save(model.state_dict(), args.save_dir + f'/best_train_model_fold_{k}.pth')
                                # torch.save(attention, args.save_dir + f'/best_attention_fold_{k}.pth')

                        scheduler.step(val_metric)
                        # log.info('Epoch: {:03d}, LR: {:7f}, '
                        #   'train Loss: {:.7f}, train acc: {:.7f},Val Loss: {:.7f}, Test Loss: {:.7f},  Val acc: {:.7f}, Test acc: {:.7f}'.format(
                        #       epoch+1, lr, train_result["Loss"], train_result["Accuracy"],val_result["Loss"], test_result["Loss"], val_result["Accuracy"], test_result["Accuracy"]))

                        log.info('Epoch: {:03d}, LR: {:7f}, \n'
                                 'train Loss: {:.7f}, train acc: {:.7f}, train recall: {:.7f}, train F1: {:.7f}, \n'
                                 'Val Loss: {:.7f}, Val acc: {:.7f}, Val recall: {:.7f}, Val F1: {:.7f}, \n'
                                 'Test Loss: {:.7f}, Test acc: {:.7f}, Test recall: {:.7f}, Test F1: {:.7f}.\n'.format(
                                    epoch + 1, lr, train_result["Loss"], train_result["Accuracy"],
                                    train_result["Recall"], train_result["F1 Score"],
                                    val_result["Loss"], val_result["Accuracy"],
                                    val_result["Recall"], val_result["F1 Score"],
                                    test_result["Loss"], test_result["Accuracy"],
                                    test_result["Recall"], test_result["F1 Score"]))

                        trl.append(train_result["Loss"])
                        tracc.append(train_result["Accuracy"])
                        vl.append(val_result["Loss"])
                        vacc.append(val_result["Accuracy"])
                        tl.append(test_result["Loss"])
                        tsacc.append(test_result["Accuracy"])

        result_for_plot = np.stack([trl, tracc, vl, vacc, tl, tsacc], axis=1)
        rfp = pd.DataFrame(result_for_plot)
        rfp.to_csv('../data/result_for_plot.csv', index=False)

        model,s=get_model(args,log,load_path=args.save_dir + f'/best_train_model_fold_{k}.pth')
        # model, s = get_model(args, log, load_path=args.load_path + f'/best_train_model_fold_{0}.pth')
        model.to(device)
        model.eval()
        # val(dataset_loader,model,device)
        train_result=val(train_loader,model,device,args.name)
        val_result=val(val_loader,model,device,args.name)
        test_result = val(test_loader, model, device,args.name)




        # log.info(f"At fold {k+1}, the best model result: "
        #          f"train Loss: {train_result['Loss']}, train Acc: {train_result['Accuracy']}, "
        #          f"Val Loss: {val_result['Loss']}, Test Loss: {test_result['Loss']}, "
        #          f" Val Acc: {val_result['Accuracy']}, Test Acc: {test_result['Accuracy']}")
        log.info(f"At fold {k + 1}, the best model result: \n"
                 f"train Loss: {train_result['Loss']}, train Acc: {train_result['Accuracy']}, "
                 f"train Recall: {train_result['Recall']}, train F1: {train_result['F1 Score']}, \n"
                 f"Val Loss: {val_result['Loss']}, Val Acc: {val_result['Accuracy']}, "
                 f"Val Recall: {val_result['Recall']}, Val F1: {val_result['F1 Score']}, \n"
                 f"Test Loss: {test_result['Loss']}, Test Acc: {test_result['Accuracy']}, "
                 f"Test Recall: {test_result['Recall']}, Test F1: {test_result['F1 Score']}.\n")

        best_val_loss[k]=val_result["Loss"]
        best_val_corr[k]=val_result["Accuracy"]
        best_test_loss[k]=test_result["Loss"]
        best_test_corr[k]=test_result["Accuracy"]
        # add recall and f1
        best_val_recall[k]=val_result["Recall"]
        best_val_f1[k]=val_result["F1 Score"]
        best_test_recall[k]=test_result["Recall"]
        best_test_f1[k]=test_result["F1 Score"]

    mean_val_loss=np.mean(best_val_loss)
    mean_test_loss=np.mean(best_test_loss)
    mean_val_corr=np.mean(best_val_corr)
    mean_test_corr=np.mean(best_test_corr)
    # add recall and f1
    mean_val_recall=np.mean(best_val_recall)
    mean_test_recall=np.mean(best_test_recall)
    mean_val_f1=np.mean(best_val_f1)
    mean_test_f1=np.mean(best_test_f1)
    log.info("-------------------------------Final Reuslt-------------------------------------")
    log.info(f"Cross validation result:"
             f"Val Loss: {mean_val_loss}, Test Loss: {mean_test_loss}, "
             f" Val Acc: {mean_val_corr}, Test Acc: {mean_test_corr}, \n"
             f"Val Recall: {mean_val_recall}, Test Recall: {mean_test_recall}, "
             f"Val F1: {mean_val_f1}, Test F1: {mean_test_f1}.")


if __name__=="__main__":
    main()





