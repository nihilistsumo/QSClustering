import math
import random

import transformers
from transformers import AdamW
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import adjusted_rand_score
from core.clustering import QuerySpecificClusteringModel, SBERTTripletLossModel
from util.data import separate_multi_batch, TRECCAR_Datset, get_similar_treccar_articles, get_article_qrels
from tqdm import tqdm, trange
import numpy as np
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


def put_features_in_device(input_features, device):
    for key in input_features.keys():
        if isinstance(input_features[key], Tensor):
            input_features[key] = input_features[key].to(device)


def is_fit_for_training(article_sample):
    if len(article_sample) < 1:
        return False
    first_batch = article_sample[0]
    q = first_batch.q
    first_batch_num_paras = len(first_batch.paras)
    return 15 < first_batch_num_paras < 150


def do_eval(dataset, model, device):
    model.eval()
    rand = 0
    num = 0
    preds = []
    for article_sample in dataset:
        for batch in article_sample:
            true_labels = batch.para_labels
            query_content = batch.q.split('enwiki:')[1].replace('%20', ' ')
            input_texts = [(query_content, t) for t in batch.para_texts]
            input_features = model.qp_model.tokenize(input_texts)
            put_features_in_device(input_features, device)
            k = len(set(batch.para_labels))
            pred_labels = model.get_clustering(input_features, k)
            preds.append(pred_labels)
            rand += adjusted_rand_score(true_labels, pred_labels)
            num += 1
    print(random.sample(preds, 1)[0])
    return rand / num


def do_triplet_eval(dataset, model):
    model.eval()
    rand = 0
    num = 0
    preds = []
    for article_sample in dataset:
        for batch in article_sample:
            true_labels = batch.para_labels
            k = len(set(batch.para_labels))
            pred_labels = model.get_clustering(batch.para_texts, k)
            preds.append(pred_labels)
            rand += adjusted_rand_score(true_labels, pred_labels)
            num += 1
    print(random.sample(preds, 1)[0])
    return rand / num


def treccar_clustering_single_model(train_dataset,
                                    test_dataset,
                                    device,
                                    val_dataset=None,
                                    max_num_tokens=128,
                                    max_grad_norm=1.0,
                                    weight_decay=0.01,
                                    warmup=10000,
                                    lrate=2e-5,
                                    num_epochs=3,
                                    emb_model_name='sentence-transformers/all-MiniLM-L6-v2',
                                    emb_dim=256,
                                    alpha=0.5):
    mse = nn.MSELoss()
    num_steps_per_epoch = len(train_dataset)
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
    val_rand = do_eval(val_dataset, model, device)
    test_rand = do_eval(test_dataset, model, device)
    print('\nVal RAND %.4f, Test RAND %.4f' % (val_rand, test_rand))
    for epoch in tqdm(range(num_epochs)):
        train_ids = list(range(len(train_dataset)))
        random.shuffle(train_ids)
        for idx in tqdm(range(len(train_ids))):
            if idx > 0 and idx % 500 == 0:
                val_rand = do_eval(val_dataset, model, device)
                test_rand = do_eval(test_dataset, model, device)
                print('\nVal RAND %.4f, Test RAND %.4f' % (val_rand, test_rand))
            model.train()
            article_sample = train_dataset[train_ids[idx]]
            if not is_fit_for_training(article_sample):
                continue
            for batch in article_sample:
                query_content = batch.q.split('enwiki:')[1].replace('%20', ' ')
                input_texts = [(query_content, t) for t in batch.para_texts]
                input_features = model.qp_model.tokenize(input_texts)
                #print(sample.q + ' max tokens %d' % input_features['input_ids'][0].size())
                gt = torch.zeros(len(batch.paras), len(batch.paras), device=device)
                gt_weights = torch.ones(len(batch.paras), len(batch.paras), device=device)
                para_label_freq = {k: batch.para_labels.count(k) for k in set(batch.para_labels)}
                for i in range(len(batch.paras)):
                    for j in range(len(batch.paras)):
                        if batch.para_labels[i] == batch.para_labels[j]:
                            gt[i][j] = 1.0
                            gt_weights[i][j] = para_label_freq[batch.para_labels[i]]
                put_features_in_device(input_features, device)
                k = len(set(batch.para_labels))
                #print(GPUtil.showUtilization())
                mc, ma = model(input_features, k)
                sim_mat = 1 / (1 + torch.cdist(ma, ma))
                #print(GPUtil.showUtilization())
                #loss = mse(gt, sim_mat)
                #reg = torch.relu(len(batch.paras) + torch.sum((1 - gt) * sim_mat) - torch.sum(gt * sim_mat))
                # weighted mse
                loss = torch.sum(((gt - sim_mat) ** 2) * gt_weights) / gt.shape[0]
                loss.backward()
                #print(batch.q + ' %d paras, Loss %.4f' % (len(batch.paras), loss.detach().item()))
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
                opt.zero_grad()
                schd.step()
        print('Evaluation')
        print('==========')
        val_rand = do_eval(val_dataset, model, device)
        print('Validation RAND %.4f' % val_rand)
        test_rand = do_eval(test_dataset, model, device)
        print('Test RAND %.4f' % test_rand)


def treccar_clustering_baseline_sbert_triplet_model(train_dataset,
                                    test_dataset,
                                    device,
                                    val_dataset=None,
                                    max_num_tokens=128,
                                    max_grad_norm=1.0,
                                    weight_decay=0.01,
                                    warmup=10000,
                                    lrate=2e-5,
                                    num_epochs=3,
                                    emb_model_name='sentence-transformers/all-MiniLM-L6-v2',
                                    triplet_margin=5,
                                    batch_size=32):
    num_steps_per_epoch = len(train_dataset)
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
    val_rand = do_triplet_eval(val_dataset, model)
    test_rand = do_triplet_eval(test_dataset, model)
    print('\nVal RAND %.4f, Test RAND %.4f' % (val_rand, test_rand))
    for epoch in tqdm(range(num_epochs)):
        train_ids = list(range(len(train_dataset)))
        random.shuffle(train_ids)
        for idx in tqdm(range(len(train_ids))):
            if idx > 0 and idx % 500 == 0:
                val_rand = do_triplet_eval(val_dataset, model)
                test_rand = do_triplet_eval(test_dataset, model)
                print('\nVal RAND %.4f, Test RAND %.4f' % (val_rand, test_rand))
            model.train()
            article_sample = train_dataset[train_ids[idx]]
            if not is_fit_for_training(article_sample):
                continue
            for batch in article_sample:
                input_texts = get_triplet_texts_from_batch(batch)
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
        print('Evaluation')
        print('==========')
        val_rand = do_triplet_eval(val_dataset, model)
        print('Validation RAND %.4f' % val_rand)
        test_rand = do_triplet_eval(test_dataset, model)
        print('Test RAND %.4f' % test_rand)


def get_triplet_texts_from_batch(batch):
    input_texts = {'anchors': [], 'pos': [], 'neg': []}
    for i in range(len(batch.paras) - 2):
        for j in range(i + 1, len(batch.paras) - 1):
            for k in range(i + 2, len(batch.paras)):
                if len({batch.para_labels[i], batch.para_labels[j], batch.para_labels[k]}) == 2:
                    if batch.para_labels[i] == batch.para_labels[j]:
                        anchor, p, n = i, j, k
                    elif batch.para_labels[i] == batch.para_labels[k]:
                        anchor, n, p = i, j, k
                    else:
                        anchor, p, n = j, k, i
                    input_texts['anchors'].append(batch.para_texts[anchor])
                    input_texts['pos'].append(batch.para_texts[p])
                    input_texts['neg'].append(batch.para_texts[n])
    return input_texts


if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available and using device: '+str(device))
else:
    device = torch.device('cpu')
    print('CUDA not available, using device: '+str(device))

train_articles = get_similar_treccar_articles(get_article_qrels('../data/train.pages.cbor-article.qrels').keys(), 'D:\\new_cats_data\\train\\base.train.cbor-without-by1-article.qrels', 100)
#train_articles = random.sample(get_article_qrels('D:\\new_cats_data\\train\\base.train.cbor-without-by1-article.qrels').keys(), 5000)

train_dataset = TRECCAR_Datset('D:\\new_cats_data\\train\\base.train.cbor-without-by1-article.qrels',
                               'D:\\new_cats_data\\train\\base.train.cbor-without-by1-toplevel.qrels',
                               'D:\\new_cats_data\\train\\train_paratext.tsv', 35, train_articles)

val_dataset = TRECCAR_Datset('../data/train.pages.cbor-article.qrels',
                             '../data/train.pages.cbor-toplevel.qrels',
                'D:\\new_cats_data\\benchmarkY1\\benchmarkY1-train-nodup\\by1train_paratext\\by1train_paratext.tsv', 35)
test_dataset = TRECCAR_Datset('../data/test.pages.cbor-article.qrels',
                              '../data/test.pages.cbor-toplevel.qrels',
                'D:\\new_cats_data\\benchmarkY1\\benchmarkY1-test-nodup\\by1test_paratext\\by1test_paratext.tsv', 35)
treccar_clustering_single_model(train_dataset, test_dataset, device, val_dataset)
#treccar_clustering_baseline_sbert_triplet_model(train_dataset, test_dataset, device, val_dataset)