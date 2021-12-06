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


def do_eval(test_data, model, max_num_tokens, qc=None):
    model.eval()
    rand_dict, nmi_dict = {}, {}
    for s in test_data.keys():
        texts = [d['abstract'] for d in test_data[s]]
        true_labels = [d['label'] for d in test_data[s]]
        k = len(set(true_labels))
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


def arxiv_clustering_single_model(arxiv_data_file,
                                    device,
                                    query_context_ref=None,
                                    loss_name='adj',
                                    num_folds=5,
                                    max_num_tokens=128,
                                    max_grad_norm=1.0,
                                    weight_decay=0.01,
                                    warmup=10000,
                                    lrate=2e-5,
                                    num_epochs=5,
                                    emb_model_name='sentence-transformers/all-MiniLM-L6-v2',
                                    emb_dim=256):
    if query_context_ref is not None:
        with open(query_context_ref, 'r') as f:
            qc = json.load(f)
    cv_datasets = prepare_arxiv_data_for_cv(arxiv_data_file, num_folds)
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
        print('Test evaluation')
        for s in test_data_current.keys():
            print(s + ' RAND %.4f, NMI %.4f' % (test_rand[s], test_nmi[s]))
        if loss_name == 'nmi':
            loss_func = get_nmi_loss
        else:
            loss_func = get_weighted_adj_rand_loss
        for epoch in tqdm(range(num_epochs)):
            for idx in tqdm(range(len(train_data_current))):
                if idx > 0 and idx % 500 == 0:
                    if query_context_ref is not None:
                        test_rand, test_nmi = do_eval(test_data_current, model, max_num_tokens, qc)
                    else:
                        test_rand, test_nmi = do_eval(test_data_current, model, max_num_tokens)
                    print('Test evaluation')
                    for s in test_data_current.keys():
                        print(s + ' RAND %.4f, NMI %.4f' % (test_rand[s], test_nmi[s]))
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
        print('Evaluation')
        print('==========')
        if query_context_ref is not None:
            test_rand, test_nmi = do_eval(test_data_current, model, max_num_tokens, qc)
        else:
            test_rand, test_nmi = do_eval(test_data_current, model, max_num_tokens)
        print('Test evaluation')
        for s in test_data_current.keys():
            print(s + ' RAND %.4f, NMI %.4f' % (test_rand[s], test_nmi[s]))


if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available and using device: '+str(device))
else:
    device = torch.device('cpu')
    print('CUDA not available, using device: '+str(device))
parser = argparse.ArgumentParser(description='Run query specific clustering experiments on Arxiv dataset')
parser.add_argument('-ad', '--arxiv_data', help='Path to arxiv clustering data json file', default='D:\\arxiv_dataset\\arxiv_clustering_data.json')

args = parser.parse_args()
arxiv_clustering_single_model(args.arxiv_data, device)