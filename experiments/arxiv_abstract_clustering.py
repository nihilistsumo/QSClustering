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
from util.data import Vital_Wiki_Dataset, get_article_qrels, get_rev_qrels, prepare_arxiv_data_for_cv
from tqdm import tqdm, trange
import json
import numpy as np
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
default_subject_desc = {'cs': 'Computer Science',
                        'econ': 'Economics',
                        'eess': 'Electrical Engineering and Systems Science',
                        'math': 'Mathematics',
                        'astro-ph': 'Astrophysics',
                        'cond-mat': 'Condensed Matter',
                        'nlin': 'Nonlinear Sciences',
                        'physics': 'Physics',
                        'q-bio': 'Quantitative Biology',
                        'q-fin': 'Quantitative Finance',
                        'stat': 'Statistics'}


def put_features_in_device(input_features, device):
    for key in input_features.keys():
        if isinstance(input_features[key], Tensor):
            input_features[key] = input_features[key].to(device)


def do_eval(test_data, model, max_num_tokens, qc=None, triplet_model=False):
    model.eval()
    rand_dict, nmi_dict = {}, {}
    for s in test_data.keys():
        true_labels = [d['label'] for d in test_data[s]]
        k = len(set(true_labels))
        if triplet_model:
            input_texts = [d['abstract'] for d in test_data[s]]
            embeddings = model.emb_model.encode(input_texts, convert_to_tensor=True)
        else:
            texts = [d['abstract'] for d in test_data[s]]
            query_content = default_subject_desc[s]
            if qc is not None:
                context = qc[s]
                query_content = ' '.join(context.split(' ')[:max_num_tokens // 2])
            input_texts = [(query_content, t) for t in texts]
            embeddings = model.qp_model.encode(input_texts, convert_to_tensor=True)
        pred_labels = model.get_clustering(embeddings, k)
        rand = adjusted_rand_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        rand_dict[s] = rand
        nmi_dict[s] = nmi
    return rand_dict, nmi_dict


def arxiv_clustering_single_model(arxiv_5cv_data_file,
                                    device,
                                    query_context_ref=None,
                                    loss_name='adj',
                                    max_num_tokens=128,
                                    max_grad_norm=1.0,
                                    weight_decay=0.01,
                                    warmup=10000,
                                    lrate=2e-5,
                                    num_epochs=2,
                                    emb_model_name='sentence-transformers/all-MiniLM-L6-v2',
                                    emb_dim=256):
    if query_context_ref is not None:
        with open(query_context_ref, 'r') as f:
            qc = json.load(f)
    cv_datasets = np.load(arxiv_5cv_data_file, allow_pickle=True)[()]['data']
    for i in range(len(cv_datasets)):
        train_data_current = cv_datasets[i]
        test_data_current = train_data_current.test_arxiv_data
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
            test_rand, test_nmi = do_eval(test_data_current, model, max_num_tokens, qc)
        else:
            test_rand, test_nmi = do_eval(test_data_current, model, max_num_tokens)
        print('\nFold %d Initial Test evaluation' % (i+1))
        for s in test_data_current.keys():
            print(s + ' RAND %.4f, NMI %.4f' % (test_rand[s], test_nmi[s]))
        if loss_name == 'nmi':
            loss_func = get_nmi_loss
        else:
            loss_func = get_weighted_adj_rand_loss
        for epoch in tqdm(range(num_epochs)):
            for idx in tqdm(range(len(train_data_current))):
                model.train()
                sample = train_data_current[idx]
                query_content = sample.q #cat
                if query_context_ref is not None:
                    context = qc[sample.q]
                    query_content = ' '.join(context.split(' ')[:max_num_tokens // 2])
                n = len(sample.papers)
                k = len(set(sample.paper_labels))
                input_texts = [(query_content, t) for t in sample.paper_texts]
                input_features = model.qp_model.tokenize(input_texts)
                put_features_in_device(input_features, device)
                #print(GPUtil.showUtilization())
                mc, ma = model(input_features, k)
                loss = loss_func(ma, sample.paper_labels, device)
                loss.backward()
                #print(batch.q + ' %d paras, Loss %.4f' % (len(batch.paras), loss.detach().item()))
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
                opt.zero_grad()
                schd.step()
        print('Evaluation Fold %d' % (i+1))
        print('=================')
        if query_context_ref is not None:
            test_rand, test_nmi = do_eval(test_data_current, model, max_num_tokens, qc)
        else:
            test_rand, test_nmi = do_eval(test_data_current, model, max_num_tokens)
        for s in test_data_current.keys():
            print(s + ' RAND %.4f, NMI %.4f' % (test_rand[s], test_nmi[s]))


def arxiv_clustering_baseline_sbert_triplet_model(arxiv_5cv_data_file,
                                    device,
                                    max_num_tokens=128,
                                    max_grad_norm=1.0,
                                    weight_decay=0.01,
                                    warmup=10000,
                                    lrate=2e-5,
                                    num_epochs=2,
                                    emb_model_name='sentence-transformers/all-MiniLM-L6-v2',
                                    triplet_margin=5,
                                    emb_dim=256,
                                    batch_size=32):
    cv_datasets = np.load(arxiv_5cv_data_file, allow_pickle=True)[()]['data']
    for i in range(len(cv_datasets)):
        train_data_current = cv_datasets[i]
        test_data_current = train_data_current.test_arxiv_data
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
        test_rand, test_nmi = do_eval(test_data_current, model, max_num_tokens, triplet_model=True)
        print('\nFold %d Initial Test evaluation' % (i + 1))
        for s in test_data_current.keys():
            print(s + ' RAND %.4f, NMI %.4f' % (test_rand[s], test_nmi[s]))
        for epoch in tqdm(range(num_epochs)):
            for idx in tqdm(range(len(train_data_current))):
                model.train()
                sample = train_data_current[idx]
                input_texts = get_triplet_texts_from_sample(sample)
                for b in range(math.ceil(len(input_texts) / batch_size)):
                    anchor_features = model.emb_model.tokenize(input_texts['anchors'][b*batch_size: (b+1)*batch_size])
                    put_features_in_device(anchor_features, device)
                    pos_features = model.emb_model.tokenize(input_texts['pos'][b*batch_size: (b+1)*batch_size])
                    put_features_in_device(pos_features, device)
                    neg_features = model.emb_model.tokenize(input_texts['neg'][b*batch_size: (b+1)*batch_size])
                    put_features_in_device(neg_features, device)
                    #print(sample.q + ' max tokens %d' % input_features['input_ids'][0].size())
                    #print(GPUtil.showUtilization())
                    loss = model(anchor_features, pos_features, neg_features)
                    loss.backward()
                    #print(batch.q + ' %d paras, Loss %.4f' % (len(batch.paras), loss.detach().item()))
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    opt.step()
                    opt.zero_grad()
                    schd.step()
        print('Evaluation Fold %d' % (i + 1))
        print('=================')
        test_rand, test_nmi = do_eval(test_data_current, model, max_num_tokens, triplet_model=True)
        for s in test_data_current.keys():
            print(s + ' RAND %.4f, NMI %.4f' % (test_rand[s], test_nmi[s]))


def get_triplet_texts_from_sample(sample):
    input_texts = {'anchors': [], 'pos': [], 'neg': []}
    for i in range(len(sample.papers) - 2):
        for j in range(i + 1, len(sample.papers) - 1):
            for k in range(i + 2, len(sample.papers)):
                if len({sample.paper_labels[i], sample.paper_labels[j], sample.paper_labels[k]}) == 2:
                    if sample.paper_labels[i] == sample.paper_labels[j]:
                        anchor, p, n = i, j, k
                    elif sample.paper_labels[i] == sample.paper_labels[k]:
                        anchor, n, p = i, j, k
                    else:
                        anchor, p, n = j, k, i
                    input_texts['anchors'].append(sample.paper_texts[anchor])
                    input_texts['pos'].append(sample.paper_texts[p])
                    input_texts['neg'].append(sample.paper_texts[n])
    return input_texts



if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available and using device: '+str(device))
else:
    device = torch.device('cpu')
    print('CUDA not available, using device: '+str(device))
parser = argparse.ArgumentParser(description='Run query specific clustering experiments on Arxiv dataset')
parser.add_argument('-ad', '--arxiv_data', help='Path to arxiv clustering npy file prepared for 5-fold cv', default='D:\\arxiv_dataset\\arxiv_clustering_data_5cv.npy')
parser.add_argument('-ne', '--experiment', type=int, help='Choose the experiment to run (1: QSC/ 2: baseline)', default=2)

args = parser.parse_args()
if args.experiment == 1:
    arxiv_clustering_single_model(args.arxiv_data, device)
elif args.experiment == 2:
    arxiv_clustering_baseline_sbert_triplet_model(args.arxiv_data, device)