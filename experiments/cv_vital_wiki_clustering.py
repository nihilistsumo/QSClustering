import math
import random
import argparse
import transformers
from transformers import AdamW
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from core.clustering import QuerySpecificClusteringModel, SBERTTripletLossModel, get_nmi_loss, \
    get_weighted_adj_rand_loss, QuerySpecificDKM
from tqdm import tqdm, trange
import json
import numpy as np
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


def put_features_in_device(input_features, device):
    for key in input_features.keys():
        if isinstance(input_features[key], Tensor):
            input_features[key] = input_features[key].to(device)


def do_eval(test_samples, model, qc=None, triplet_model=False):
    model.eval()
    rand_dict, nmi_dict = {}, {}
    for s in test_samples:
        true_labels = s.para_labels
        k = len(set(true_labels))
        if triplet_model:
            input_texts = s.para_texts
            embeddings = model.emb_model.encode(input_texts, convert_to_tensor=True)
        else:
            texts = s.para_texts
            query_content = s.category +'. ' + s.q.split('enwiki:')[1].replace('%20', ' ')
            input_texts = [(query_content, t) for t in texts]
            embeddings = model.qp_model.encode(input_texts, convert_to_tensor=True)
        pred_labels = model.get_clustering(embeddings, k)
        rand = adjusted_rand_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        rand_dict[s.q] = rand
        nmi_dict[s.q] = nmi
    return rand_dict, nmi_dict


def do_eval_dkm_param(test_samples, model, qc=None):
    model.eval()
    rand_dict, nmi_dict = {}, {}
    for s in test_samples:
        true_labels = s.para_labels
        k = len(set(true_labels))
        texts = s.para_texts
        query_content = s.category +'. ' + s.q.split('enwiki:')[1].replace('%20', ' ')
        pred_labels = model.get_clustering(query_content, texts, k)
        rand = adjusted_rand_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        rand_dict[s.q] = rand
        nmi_dict[s.q] = nmi
    return rand_dict, nmi_dict


def vital_wiki_clustering_single_model(vital_wiki_2cv_data_file,
                                    device,
                                    loss_name,
                                    query_context_ref=None,
                                    max_num_tokens=128,
                                    max_grad_norm=1.0,
                                    weight_decay=0.01,
                                    warmup=1000,
                                    lrate=2e-5,
                                    num_epochs=2,
                                    emb_model_name='sentence-transformers/all-MiniLM-L6-v2',
                                    emb_dim=256):
    if query_context_ref is not None:
        with open(query_context_ref, 'r') as f:
            qc = json.load(f)
    cv_datasets = np.load(vital_wiki_2cv_data_file, allow_pickle=True)[()]['data']
    for i in range(len(cv_datasets)):
        train_data_current = cv_datasets[i]
        test_data_current = train_data_current.test_samples
        num_steps_per_epoch = len(train_data_current)
        num_train_steps = num_epochs * num_steps_per_epoch
        model = QuerySpecificClusteringModel(emb_model_name, emb_dim, device, max_num_tokens)
        model_params = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model_params if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay},
            {'params': [p for n, p in model_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        opt = AdamW(optimizer_grouped_parameters, lr=lrate)
        schd = transformers.get_linear_schedule_with_warmup(opt, warmup, num_epochs * num_train_steps)
        if query_context_ref is not None:
            test_rand, test_nmi = do_eval(test_data_current, model, qc)
        else:
            test_rand, test_nmi = do_eval(test_data_current, model)
        print('\nFold %d Initial Test evaluation' % (i+1))
        print('Mean RAND %.4f +- %.4f, NMI %.4f +- %.4f' % (np.mean(list(test_rand.values())),
                                                            np.std(list(test_rand.values()), ddof=1) / np.sqrt(len(test_rand.keys())),
                                                            np.mean(list(test_nmi.values())),
                                                            np.std(list(test_nmi.values()), ddof=1) / np.sqrt(len(test_nmi.keys()))))
        if loss_name == 'nmi':
            loss_func = get_nmi_loss
        else:
            loss_func = get_weighted_adj_rand_loss
        for epoch in tqdm(range(num_epochs)):
            for idx in tqdm(range(len(train_data_current))):
                model.train()
                sample = train_data_current[idx]
                query_content = sample.category +'. ' + sample.q.split('enwiki:')[1].replace('%20', ' ')
                n = len(sample.paras)
                k = len(set(sample.para_labels))
                input_texts = [(query_content, t) for t in sample.para_texts]
                input_features = model.qp_model.tokenize(input_texts)
                put_features_in_device(input_features, device)
                #print(GPUtil.showUtilization())
                mc, ma = model(input_features, k)
                loss = loss_func(ma, sample.para_labels, device)
                loss.backward()
                #print(batch.q + ' %d paras, Loss %.4f' % (len(batch.paras), loss.detach().item()))
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
                opt.zero_grad()
                schd.step()
        print('Evaluation Fold %d' % (i+1))
        print('=================')
        if query_context_ref is not None:
            test_rand, test_nmi = do_eval(test_data_current, model, qc)
        else:
            test_rand, test_nmi = do_eval(test_data_current, model)
        print('Mean RAND %.4f +- %.4f, NMI %.4f +- %.4f' % (np.mean(list(test_rand.values())),
                                                            np.std(list(test_rand.values()), ddof=1) / np.sqrt(len(test_rand.keys())),
                                                            np.mean(list(test_nmi.values())),
                                                            np.std(list(test_nmi.values()), ddof=1) / np.sqrt(len(test_nmi.keys()))))


def vital_wiki_clustering_dkm_param_model(vital_wiki_2cv_data_file,
                                    device,
                                    loss_name,
                                    query_context_ref=None,
                                    max_num_tokens=128,
                                    max_grad_norm=1.0,
                                    weight_decay=0.01,
                                    warmup=10000,
                                    lrate=2e-5,
                                    num_epochs=50,
                                    emb_model_name='sentence-transformers/all-MiniLM-L6-v2',
                                    emb_dim=256):
    if query_context_ref is not None:
        with open(query_context_ref, 'r') as f:
            qc = json.load(f)
    cv_datasets = np.load(vital_wiki_2cv_data_file, allow_pickle=True)[()]['data']
    for i in range(len(cv_datasets)):
        train_data_current = cv_datasets[i]
        test_data_current = train_data_current.test_samples
        num_steps_per_epoch = len(train_data_current)
        num_train_steps = num_epochs * num_steps_per_epoch
        model = QuerySpecificDKM(emb_model_name, emb_dim, device, max_num_tokens)
        model_params = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model_params if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay},
            {'params': [p for n, p in model_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        opt = AdamW(optimizer_grouped_parameters, lr=lrate)
        schd = transformers.get_linear_schedule_with_warmup(opt, warmup, num_epochs * num_train_steps)
        if query_context_ref is not None:
            test_rand, test_nmi = do_eval_dkm_param(test_data_current, model, qc)
        else:
            test_rand, test_nmi = do_eval_dkm_param(test_data_current, model)
        print('\nFold %d Initial Test evaluation' % (i+1))
        print('Mean RAND %.4f +- %.4f, NMI %.4f +- %.4f' % (np.mean(list(test_rand.values())),
                                                            np.std(list(test_rand.values()), ddof=1) / np.sqrt(len(test_rand.keys())),
                                                            np.mean(list(test_nmi.values())),
                                                            np.std(list(test_nmi.values()), ddof=1) / np.sqrt(len(test_nmi.keys()))))
        if loss_name == 'nmi':
            loss_func = get_nmi_loss
        else:
            loss_func = get_weighted_adj_rand_loss
        for epoch in tqdm(range(num_epochs)):
            for idx in tqdm(range(len(train_data_current))):
                model.train()
                sample = train_data_current[idx]
                query_content = sample.category +'. ' + sample.q.split('enwiki:')[1].replace('%20', ' ')
                n = len(sample.paras)
                k = len(set(sample.para_labels))
                #print(GPUtil.showUtilization())
                mc, ma = model(query_content, sample.para_texts, k)
                loss = loss_func(ma, sample.para_labels, device)
                loss.backward()
                #print(batch.q + ' %d paras, Loss %.4f' % (len(batch.paras), loss.detach().item()))
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
                opt.zero_grad()
                schd.step()
        print('Evaluation Fold %d' % (i+1))
        print('=================')
        if query_context_ref is not None:
            test_rand, test_nmi = do_eval_dkm_param(test_data_current, model, qc)
        else:
            test_rand, test_nmi = do_eval_dkm_param(test_data_current, model)
        print('Mean RAND %.4f +- %.4f, NMI %.4f +- %.4f' % (np.mean(list(test_rand.values())),
                                                            np.std(list(test_rand.values()), ddof=1) / np.sqrt(len(test_rand.keys())),
                                                            np.mean(list(test_nmi.values())),
                                                            np.std(list(test_nmi.values()), ddof=1) / np.sqrt(len(test_nmi.keys()))))


def vital_wiki_clustering_baseline_sbert_triplet_model(vital_wiki_2cv_data_file,
                                    device,
                                    max_num_tokens=128,
                                    max_grad_norm=1.0,
                                    weight_decay=0.01,
                                    warmup=1000,
                                    lrate=2e-5,
                                    num_epochs=2,
                                    emb_model_name='sentence-transformers/all-MiniLM-L6-v2',
                                    triplet_margin=5,
                                    emb_dim=256,
                                    batch_size=32):
    cv_datasets = np.load(vital_wiki_2cv_data_file, allow_pickle=True)[()]['data']
    for i in range(len(cv_datasets)):
        train_data_current = cv_datasets[i]
        test_data_current = train_data_current.test_samples
        num_steps_per_epoch = len(train_data_current)
        num_train_steps = num_epochs * num_steps_per_epoch
        model = SBERTTripletLossModel(emb_model_name, device, max_num_tokens, triplet_margin)
        model_params = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model_params if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay},
            {'params': [p for n, p in model_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        opt = AdamW(optimizer_grouped_parameters, lr=lrate)
        schd = transformers.get_linear_schedule_with_warmup(opt, warmup, num_epochs * num_train_steps)
        test_rand, test_nmi = do_eval(test_data_current, model, triplet_model=True)
        print('\nFold %d Initial Test evaluation' % (i+1))
        print('Mean RAND %.4f +- %.4f, NMI %.4f +- %.4f' % (np.mean(list(test_rand.values())),
                                                            np.std(list(test_rand.values()), ddof=1) / np.sqrt(len(test_rand.keys())),
                                                            np.mean(list(test_nmi.values())),
                                                            np.std(list(test_nmi.values()), ddof=1) / np.sqrt(len(test_nmi.keys()))))
        for epoch in tqdm(range(num_epochs)):
            for idx in tqdm(range(len(train_data_current))):
                model.train()
                sample = train_data_current[idx]
                input_texts = get_triplet_texts_from_sample(sample)
                for b in range(math.ceil(len(input_texts) / batch_size)):
                    anchor_features = model.emb_model.tokenize(input_texts['anchors'][b * batch_size: (b + 1) * batch_size])
                    put_features_in_device(anchor_features, device)
                    pos_features = model.emb_model.tokenize(input_texts['pos'][b * batch_size: (b + 1) * batch_size])
                    put_features_in_device(pos_features, device)
                    neg_features = model.emb_model.tokenize(input_texts['neg'][b * batch_size: (b + 1) * batch_size])
                    put_features_in_device(neg_features, device)
                    # print(sample.q + ' max tokens %d' % input_features['input_ids'][0].size())
                    # print(GPUtil.showUtilization())
                    loss = model(anchor_features, pos_features, neg_features)
                    loss.backward()
                    # print(batch.q + ' %d paras, Loss %.4f' % (len(batch.paras), loss.detach().item()))
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    opt.step()
                    opt.zero_grad()
                    schd.step()
        print('Evaluation Fold %d' % (i+1))
        print('=================')
        test_rand, test_nmi = do_eval(test_data_current, model, triplet_model=True)
        print('Mean RAND %.4f +- %.4f, NMI %.4f +- %.4f' % (np.mean(list(test_rand.values())),
                                                            np.std(list(test_rand.values()), ddof=1) / np.sqrt(len(test_rand.keys())),
                                                            np.mean(list(test_nmi.values())),
                                                            np.std(list(test_nmi.values()), ddof=1) / np.sqrt(len(test_nmi.keys()))))


def get_triplet_texts_from_sample(sample):
    input_texts = {'anchors': [], 'pos': [], 'neg': []}
    for i in range(len(sample.paras) - 2):
        for j in range(i + 1, len(sample.paras) - 1):
            for k in range(i + 2, len(sample.paras)):
                if len({sample.para_labels[i], sample.para_labels[j], sample.para_labels[k]}) == 2:
                    if sample.para_labels[i] == sample.para_labels[j]:
                        anchor, p, n = i, j, k
                    elif sample.para_labels[i] == sample.para_labels[k]:
                        anchor, n, p = i, j, k
                    else:
                        anchor, p, n = j, k, i
                    input_texts['anchors'].append(sample.para_texts[anchor])
                    input_texts['pos'].append(sample.para_texts[p])
                    input_texts['neg'].append(sample.para_texts[n])
    return input_texts


if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available and using device: '+str(device))
else:
    device = torch.device('cpu')
    print('CUDA not available, using device: '+str(device))
parser = argparse.ArgumentParser(description='Run query specific clustering experiments on Vital wiki 2-fold cv dataset')
parser.add_argument('-vd', '--vital_data', help='Path to vital wiki clustering npy file prepared for 2-fold cv',
                    default='D:\\new_cats_data\\QSC_data\\vital_wiki\\vital_wiki_clustering_data_2cv.npy')
parser.add_argument('-ne', '--experiment', type=int, help='Choose the experiment to run (1: QSC/ 2: DKM param/ 3: baseline)', default=2)
parser.add_argument('-ls', '--loss', help='Loss func to use for QSC', default='adj')

args = parser.parse_args()
if args.experiment == 1:
    vital_wiki_clustering_single_model(args.vital_data, device, args.loss)
elif args.experiment == 2:
    vital_wiki_clustering_dkm_param_model(args.vital_data, device, args.loss)
elif args.experiment == 3:
    vital_wiki_clustering_baseline_sbert_triplet_model(args.vital_data, device)