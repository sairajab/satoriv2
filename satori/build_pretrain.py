import numpy as np
import os
import pickle
# import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, MultiStepLR, StepLR
from fastprogress import progress_bar
from sklearn import metrics
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from utils import get_shuffled_background, get_shuffled_background_presaved

# local imports
from datasets import DatasetLoadAll, DatasetLazyLoad
from extract_motifs import get_motif
from utils import get_shuffled_background
from cnn_weights import *
from sei_model import Sei
from experiment import get_indices, get_results_for_shuffled, motif_analysis
import torch
from torch.utils.tensorboard import SummaryWriter

from pre_train_model import PretrainCNN


class argspace:
    def __init__(self, numLabels, dbpath, background, directory, tomtompath='/s/chromatin/p/nobackup/Saira/meme/src/tomtom'):
        self.numLabels = numLabels
        self.tomtomPath = tomtompath
        self.tfDatabase = dbpath
        self.directory = directory
        self.intBackground = background
        self.storeInterCNN = False
        self.scoreCutoff = 0.65
        self.tomtomDist = 'ed'
        self.tomtomPval = 0.05
        self.load_motif_weights = False


def load_data(device, input_prefix, output_dir, dataset, num_labels):
    batchSize = 64
    numWorkers = 8

    test_split = 0.1
    print(input_prefix, num_labels)
    final_dataset = DatasetLazyLoad(input_prefix, num_labels=num_labels)

    seq_len = final_dataset.get_seq_len()
    train_indices, test_indices, valid_indices = get_indices(
        len(final_dataset), test_split, output_dir, dataset, "train")
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)
    train_loader = DataLoader(
        final_dataset, batch_size=batchSize, sampler=train_sampler, num_workers=numWorkers)
    test_loader = DataLoader(
        final_dataset, batch_size=batchSize, sampler=test_sampler, num_workers=numWorkers)
    valid_loader = DataLoader(
        final_dataset, batch_size=batchSize, sampler=valid_sampler, num_workers=numWorkers)

    return train_loader, valid_loader, test_loader


def run_pretraining(device, input_prefix, output_dir, dataset, num_labels):
    writer = SummaryWriter("../pre_train_logs/")
    batchSize = 128
    numWorkers = 8
    max_epochs = 100

    # Loading dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_loader, valid_loader, test_loader = load_data(
        device, input_prefix, output_dir, dataset, num_labels)

    # Loading model
    # try this too,  Sei().to(device)  #
    model = Sei().to(device)

    if num_labels == 2:
        criterion = nn.CrossEntropyLoss(reduction='mean')
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    else:
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001,
                              momentum=0.9, weight_decay=0.001)

        # optimizer = optim.Adam(model.parameters())

    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    logs = open(output_dir + "/training_logs.txt", "w")

    saved_model_dir = output_dir+'/Saved_Model'

    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir)
    else:
        try:
            checkpoint = torch.load(saved_model_dir+'/model')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print("Loaded model for finetunig...", loss)
        except:
            raise Exception(
                f"No pre-trained model found  for finetuning at {saved_model_dir}! Please run with --mode set to train.")

    optimizer.param_groups[0]['lr'] = 0.005
    # Main train/test loop
    best_valid_loss = np.inf
    best_valid_auc = np.inf

    for epoch in range(1, max_epochs + 1):
        print("Epoch : ", epoch)
        model.train()
        running_loss = 0.0
        train_auc = []
        all_labels = []
        all_preds = []

        for batch_idx, (headers, seqs, data, target) in enumerate(train_loader):
            data, target = data.to(device, dtype=torch.float), target.to(
                device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(data)
            labels = target.cpu().numpy()
            if num_labels == 2:
                loss = criterion(outputs, target)
                softmax = torch.nn.Softmax(dim=1)
                pred = softmax(outputs).cpu().detach().numpy()
                try:
                    train_auc.append(metrics.roc_auc_score(labels, pred[:, 1]))
                except:
                    train_auc.append(0.0)
            else:
                loss = criterion(outputs, target.float())
                sigmoid = torch.nn.Sigmoid()
                pred = sigmoid(outputs).cpu().detach().numpy()
                all_labels += labels.tolist()
                all_preds += pred.tolist()

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        for j in range(0, len(all_labels[0])):
            cls_labels = np.asarray(all_labels)[:, j]
            pred_probs = np.asarray(all_preds)[:, j]
            auc_score = metrics.roc_auc_score(
                cls_labels.astype(int), pred_probs)
            train_auc.append(auc_score)

        train_loss = running_loss / len(train_loader)
        print("Train Results (Loss and AUC): ",
              train_loss, np.asarray(train_auc).mean())

        # Valid
        running_loss = 0.0
        model.eval()
        valid_auc = []
        with torch.no_grad():
            for batch_idx, (headers, seqs, data, target) in enumerate(valid_loader):
                data, target = data.to(device, dtype=torch.float), target.to(
                    device, dtype=torch.long)
                outputs = model(data)
                labels = target.cpu().numpy()

                if num_labels == 2:
                    loss = criterion(outputs, target)
                    softmax = torch.nn.Softmax(dim=1)
                    pred = softmax(outputs)
                    pred = pred.cpu().detach().numpy()
                    label_pred = np.column_stack((labels, pred[:, 1]))
                    # roc = np.row_stack((roc, label_pred))
                    try:
                        valid_auc.append(
                            metrics.roc_auc_score(labels, pred[:, 1]))
                    except:
                        valid_auc.append(0.0)
                else:
                    loss = criterion(outputs, target.float())
                    sigmoid = torch.nn.Sigmoid()
                    pred = sigmoid(outputs).cpu().detach().numpy()
                    all_labels += labels.tolist()
                    all_preds += pred.tolist()

                running_loss += loss.item()

            for j in range(0, len(all_labels[0])):
                cls_labels = np.asarray(all_labels)[:, j]
                pred_probs = np.asarray(all_preds)[:, j]
                auc_score = metrics.roc_auc_score(
                    cls_labels.astype(int), pred_probs)
                valid_auc.append(auc_score)

            valid_loss = running_loss / len(valid_loader)
            res_valid_auc = np.mean(valid_auc)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_valid_auc = res_valid_auc
                print("Best Validation Loss: %.3f and AUC: %.2f" %
                      (best_valid_loss, best_valid_auc), "\n")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': valid_loss
                }, saved_model_dir + '/model')
                best_model = model

        scheduler.step(valid_loss)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", valid_loss, epoch)
        print('Epoch-{0} lr: {1} valid_loss: {2}'.format(epoch,
              optimizer.param_groups[0]['lr'], valid_loss))
        logs.writelines('Epoch-{0} lr: {1} valid_loss: {2} \n'.format(
            epoch, optimizer.param_groups[0]['lr'], valid_loss))

    return best_model, test_loader


def test_model(model, test_loader, num_labels, device, output_dir, motifs_layer=False):
    if num_labels == 2:
        criterion = nn.CrossEntropyLoss(reduction='mean')
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    else:
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.0001)

        # optimizer = optim.Adam(model.parameters())
    # Test
    model.eval()

    if motifs_layer:
        # first conv layer without the maxpooling part
        CNNlayer = model.cnn_module.motif_scanning_layer[0:3]
        CNNlayer.eval()
    all_labels = []
    all_preds = []
    running_loss = 0.0
    per_batch_labelPreds = {}
    per_batch_CNNoutput = {}
    per_batch_testSeqs = {}
    roc = np.asarray([[], []]).T
    CNNWeights = []
    test_auc = []
    running_loss = 0
    with torch.no_grad():
        for batch_idx, (headers, seqs, data, target) in enumerate(test_loader):
            data, target = data.to(device, dtype=torch.float), target.to(
                device, dtype=torch.long)
            outputs = model(data)
            if motifs_layer:
                outputCNN = CNNlayer(data)
            labels = target.cpu().numpy()

            if num_labels == 2:
                loss = criterion(outputs, target)
                softmax = torch.nn.Softmax(dim=1)
                pred = softmax(outputs)
                pred = pred.cpu().detach().numpy()
                label_pred = np.column_stack((labels, pred[:, 1]))
                roc = np.row_stack((roc, label_pred))

                try:
                    test_auc.append(metrics.roc_auc_score(labels, pred[:, 1]))
                except:
                    test_auc.append(0.0)
            else:
                loss = criterion(outputs, target.float())
                sigmoid = torch.nn.Sigmoid()
                pred = sigmoid(outputs).cpu().detach().numpy()
                all_labels += labels.tolist()
                all_preds += pred.tolist()
                label_pred = {'labels': labels, 'preds': pred}

            per_batch_labelPreds[batch_idx] = label_pred
            if motifs_layer:
                per_batch_CNNoutput[batch_idx] = outputCNN.cpu(
                ).detach().numpy()
            per_batch_testSeqs[batch_idx] = np.column_stack((headers, seqs))
            running_loss += loss.item()

        for j in range(0, len(all_labels[0])):
            cls_labels = np.asarray(all_labels)[:, j]
            pred_probs = np.asarray(all_preds)[:, j]
            auc_score = metrics.roc_auc_score(
                cls_labels.astype(int), pred_probs)
            test_auc.append(auc_score)
        test_loss = running_loss / len(test_loader)
        res_test_auc = np.mean(test_auc)
        print("Test Loss: %.3f and AUC: %.2f" %
              (test_loss, res_test_auc), "\n")
        PAttn_all = {}
        res_blob = test_loss, res_test_auc, roc, PAttn_all, per_batch_labelPreds, per_batch_CNNoutput, per_batch_testSeqs
        if num_labels == 2:
            labels = roc[:, 0]
            preds = roc[:, 1]
            auprc_valid = metrics.average_precision_score(labels, preds)
            some_res = [['Test_Loss', 'Test_AUC', 'Test_AUPRC']]
            some_res.append([test_loss, res_test_auc, auprc_valid])
            np.savetxt(output_dir+'/test_results.txt',
                       some_res, fmt='%s', delimiter='\t')
        else:
            np.savetxt(output_dir+'/per_class_AUC.txt',
                       test_auc, delimiter='\t', fmt='%s')
            np.savetxt(output_dir+'/median_mean_AUC.txt',
                   np.array([np.median(test_auc) , np.mean(test_auc)]), delimiter='\t', fmt='%s')


        if motifs_layer:
            CNNWeights = model.cnn_module.motif_scanning_layer[0].weight.cpu(
            ).detach().numpy()

    return res_blob, CNNWeights


def motif_analysis_mc(model, res_test, CNN_weights, num_labels, input_prefix, batch_size, arg_space, params_dict):

    motif_dir_pos, _ = motif_analysis(
        test_resBlob, CNNWeights, arg_space, params_dict)

    bg_prefix = get_shuffled_background_presaved(input_prefix)
    data_bg = DatasetLazyLoad(bg_prefix, num_labels)
    test_loader_bg = DataLoader(
        data_bg, batch_size=batchSize, num_workers=argSpace.numWorkers)

    res_test_bg = test_model(model, test_loader, num_labels, device)[0]
    motif_dir_neg, _ = motif_analysis(
        res_test_bg, CNNWeights, arg_space, params_dict, for_background=True)


def motif_analysis_binary(res_test, CNN_weights, num_labels, arg_space, params_dict):

    test_resBlob = res_test
    CNNWeights = CNN_weights

    # print(test_resBlob)
    motif_dir_pos, _ = motif_analysis(
        test_resBlob, CNNWeights, arg_space, params_dict)

    motif_dir_neg, _ = motif_analysis(
        test_resBlob, CNNWeights,  arg_space, params_dict, for_background=True)


def load_model(path, num_labels):
    net = Sei().to(device)  # PretrainCNN(6, num_labels)
    if num_labels == 2:
        criterion = nn.CrossEntropyLoss(reduction='mean')
        optimizer = optim.SGD(net.parameters(), lr=0.01)
    else:
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.0001)

    try:
        checkpoint = torch.load(path+'/model')
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return net

    except:
        raise Exception(
            f"No pre-trained model found  for testing at {path}! Please run with --mode set to train.")


if __name__ == "__main__":

    # data_prefix = "/s/chromatin/p/nobackup/Saira/original/satori/data/ToyData/NEWDATA/ctf_40pairs_eq"
    # outdir = "/s/chromatin/p/nobackup/Saira/original/satori/results/pretrain/data-40/"
    # device = "cuda"
    # num_labels = 2
    # dataset = "simulated"
    # dbpath = "../create_dataset/subset60.meme"
    # background = "negative"
    # params = {}
    # argspace_vars = argspace(num_labels, dbpath, background, outdir)
    # _, _, test_loader = load_data(
    #     device, data_prefix, outdir, dataset, num_labels)
    # # run_pretraining(device, data_prefix, outdir, dataset, num_labels)
    # model = load_model(outdir + "/Saved_Model/", num_labels)
    # print(model)
    # test_res, CNN_weights = test_model(
    #     model, test_loader, num_labels, device, outdir)
    # motif_analysis_binary(test_res, CNN_weights,
    #                       num_labels, argspace_vars, params)

    '''
    Arabidpsis 
    '''
    # data_prefix = "/s/chromatin/p/nobackup/Saira/original/satori/data/Arabidopsis_ChromAccessibility/atAll_m200_s600"
    # outdir = "/s/chromatin/p/nobackup/Saira/original/satori/results/pretrain/sei_wd3/"
    # device = "cuda"
    # num_labels = 36
    # dataset = "arabidopsis"
    # dbpath = "/s/chromatin/p/nobackup/Saira/motif_databases/ArabidopsisDAPv1.meme"
    # background = "shuffle"
    # params = {}
    # argspace_vars = argspace(num_labels, dbpath, background, outdir)

    # model, test_loader = run_pretraining(
    #     device, data_prefix, outdir, dataset, num_labels)
    # # _, _, test_loader = load_data(
    # #     device, data_prefix, outdir, dataset, num_labels)
    # # model = load_model(outdir + "/Saved_Model/", num_labels)
    # print(model)
    # test_res, CNN_weights = test_model(
    #     model, test_loader, num_labels, device, outdir)
    # # motif_analysis_mc(test_res, CNN_weights,
    #                   num_labels, argspace_vars, params)

    data_prefix = "/s/chromatin/p/nobackup/Saira/original/satori/data/Human_Promoters/encode_roadmap_inPromoter"
    outdir = "/s/chromatin/p/nobackup/Saira/original/satori/results/pretrain/human_promoters/sei/"
    device = "cuda"
    num_labels = 164
    dataset = "human_promoters"
    dbpath = "/s/chromatin/p/nobackup/Saira/motif_databases/Homo_sapiens.meme"
    background = "shuffle"
    params = {}
    argspace_vars = argspace(num_labels, dbpath, background, outdir)

    model, test_loader = run_pretraining(
        device, data_prefix, outdir, dataset, num_labels)
    # # _, _, test_loader = load_data(
    # #     device, data_prefix, outdir, dataset, num_labels)
    # model = load_model(outdir + "/Saved_Model/", num_labels)
    # print(model)
    test_res, CNN_weights = test_model(
        model, test_loader, num_labels, device, outdir)
    # motif_analysis_mc(test_res, CNN_weights,
    #                   num_labels, argspace_vars, params)

