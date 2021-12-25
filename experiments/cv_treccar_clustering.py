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
    get_weighted_adj_rand_loss, QuerySpecificDKM, QuerySpecificClusteringModelWithSection
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


def do_eval(test_samples, model, qc=None, triplet_model=False, no_query=False):
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
            query_content = s.q.split('enwiki:')[1].replace('%20', ' ')
            if no_query:
                input_texts = texts
            else:
                input_texts = [(query_content, t) for t in texts]
            embeddings = model.qp_model.encode(input_texts, convert_to_tensor=True)
        pred_labels = model.get_clustering(embeddings, k)
        rand = adjusted_rand_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        rand_dict[s.q] = rand
        nmi_dict[s.q] = nmi
    return rand_dict, nmi_dict


def treccar_clustering_single_model(treccar_2cv_data_file,
                                    device,
                                    loss_name,
                                    query_context_ref,
                                    max_num_tokens,
                                    max_grad_norm,
                                    weight_decay,
                                    warmup,
                                    lrate,
                                    num_epochs,
                                    emb_model_name,
                                    emb_dim,
                                    output_path):
    if query_context_ref is not None:
        with open(query_context_ref, 'r') as f:
            qc = json.load(f)
    cv_datasets = np.load(treccar_2cv_data_file, allow_pickle=True)[()]['data']
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
            running_loss = 0
            for idx in tqdm(range(len(train_data_current))):
                model.train()
                sample = train_data_current[idx]
                query_content = sample.q.split('enwiki:')[1].replace('%20', ' ')
                n = len(sample.paras)
                k = len(set(sample.para_labels))
                input_texts = [(query_content, t) for t in sample.para_texts]
                input_features = model.qp_model.tokenize(input_texts)
                put_features_in_device(input_features, device)
                #print(GPUtil.showUtilization())
                mc, ma = model(input_features, k)
                loss = loss_func(ma, sample.para_labels, device)
                loss.backward()
                running_loss += loss.item()
                #print(batch.q + ' %d paras, Loss %.4f' % (len(batch.paras), loss.detach().item()))
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
                opt.zero_grad()
                schd.step()
            if query_context_ref is not None:
                test_rand, test_nmi = do_eval(test_data_current, model, qc)
            else:
                test_rand, test_nmi = do_eval(test_data_current, model)
            print('Epoch %d, mean loss: %.4f, mean RAND %.4f +- %.4f, mean NMI %.4f +- %.4f' % (epoch + 1,
                                                                                                running_loss / len(train_data_current),
                                                                                                np.mean(list(test_rand.values())),
                                                                                                np.std(list(test_rand.values()), ddof=1) / np.sqrt(len(test_rand.keys())),
                                                                                                np.mean(list(test_nmi.values())),
                                                                                                np.std(list(test_nmi.values()), ddof=1) / np.sqrt(len(test_nmi.keys()))))
        if output_path is not None:
            print('Saving the trained model...')
            torch.save(model.state_dict(), output_path+'_fold'+str(i+1)+'.model')
            model = QuerySpecificClusteringModel(emb_model_name, emb_dim, device, max_num_tokens)
            model.load_state_dict(torch.load(output_path+'_fold'+str(i+1)+'.model'))
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


def treccar_clustering_single_model_full_train(treccar_full_data_file,
                                    no_query_mode,
                                    device,
                                    loss_name,
                                    query_context_ref,
                                    max_num_tokens,
                                    max_grad_norm,
                                    weight_decay,
                                    warmup,
                                    lrate,
                                    num_epochs,
                                    emb_model_name,
                                    emb_dim,
                                    output_path):
    if query_context_ref is not None:
        with open(query_context_ref, 'r') as f:
            qc = json.load(f)
    treccar_dataset = np.load(treccar_full_data_file, allow_pickle=True)[()]['data']
    val_samples = treccar_dataset.val_samples
    test_samples = treccar_dataset.test_samples
    num_steps_per_epoch = len(treccar_dataset)
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
        val_rand, val_nmi = do_eval(val_samples, model, qc)
        test_rand, test_nmi = do_eval(test_samples, model, qc)
    else:
        val_rand, val_nmi = do_eval(val_samples, model, no_query=no_query_mode)
        test_rand, test_nmi = do_eval(test_samples, model, no_query=no_query_mode)
    print('\nInitial evaluation')
    print('Mean Val RAND %.4f +- %.4f, Val NMI %.4f +- %.4f Test RAND %.4f +- %.4f, Test NMI %.4f +- %.4f' % (
        np.mean(list(val_rand.values())),
        np.std(list(val_rand.values()), ddof=1) / np.sqrt(len(val_rand.keys())),
        np.mean(list(val_nmi.values())),
        np.std(list(val_nmi.values()), ddof=1) / np.sqrt(len(val_nmi.keys())),
        np.mean(list(test_rand.values())),
        np.std(list(test_rand.values()), ddof=1) / np.sqrt(len(test_rand.keys())),
        np.mean(list(test_nmi.values())),
        np.std(list(test_nmi.values()), ddof=1) / np.sqrt(len(test_nmi.keys()))
    ))
    if loss_name == 'nmi':
        loss_func = get_nmi_loss
    else:
        loss_func = get_weighted_adj_rand_loss
    for epoch in tqdm(range(num_epochs)):
        for idx in tqdm(range(len(treccar_dataset))):
            model.train()
            sample = treccar_dataset[idx]
            query_content = sample.q.split('enwiki:')[1].replace('%20', ' ')
            n = len(sample.paras)
            k = len(set(sample.para_labels))
            if no_query_mode:
                input_texts = sample.para_texts
            else:
                input_texts = [(query_content, t) for t in sample.para_texts]
            input_features = model.qp_model.tokenize(input_texts)
            put_features_in_device(input_features, device)
            # print(GPUtil.showUtilization())
            mc, ma = model(input_features, k)
            loss = loss_func(ma, sample.para_labels, device)
            loss.backward()
            # print(batch.q + ' %d paras, Loss %.4f' % (len(batch.paras), loss.detach().item()))
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            opt.zero_grad()
            schd.step()
            if (idx + 1) % 10000 == 0:
                if query_context_ref is not None:
                    val_rand, val_nmi = do_eval(val_samples, model, qc)
                    test_rand, test_nmi = do_eval(test_samples, model, qc)
                else:
                    val_rand, val_nmi = do_eval(val_samples, model, no_query=no_query_mode)
                    test_rand, test_nmi = do_eval(test_samples, model, no_query=no_query_mode)
                print(
                    'Mean Val RAND %.4f +- %.4f, Val NMI %.4f +- %.4f Test RAND %.4f +- %.4f, Test NMI %.4f +- %.4f' % (
                        np.mean(list(val_rand.values())),
                        np.std(list(val_rand.values()), ddof=1) / np.sqrt(len(val_rand.keys())),
                        np.mean(list(val_nmi.values())),
                        np.std(list(val_nmi.values()), ddof=1) / np.sqrt(len(val_nmi.keys())),
                        np.mean(list(test_rand.values())),
                        np.std(list(test_rand.values()), ddof=1) / np.sqrt(len(test_rand.keys())),
                        np.mean(list(test_nmi.values())),
                        np.std(list(test_nmi.values()), ddof=1) / np.sqrt(len(test_nmi.keys()))
                ))
    if output_path is not None:
        print('Saving the trained model...')
        torch.save(model.state_dict(), output_path + '.model')
        model = QuerySpecificClusteringModel(emb_model_name, emb_dim, device, max_num_tokens)
        model.load_state_dict(torch.load(output_path + '.model'))
    if query_context_ref is not None:
        val_rand, val_nmi = do_eval(val_samples, model, qc)
        test_rand, test_nmi = do_eval(test_samples, model, qc)
    else:
        val_rand, val_nmi = do_eval(val_samples, model, no_query=no_query_mode)
        test_rand, test_nmi = do_eval(test_samples, model, no_query=no_query_mode)
    print('Final Evaluation')
    print('================')
    print('Mean Val RAND %.4f +- %.4f, Val NMI %.4f +- %.4f Test RAND %.4f +- %.4f, Test NMI %.4f +- %.4f' % (
                np.mean(list(val_rand.values())),
                np.std(list(val_rand.values()), ddof=1) / np.sqrt(len(val_rand.keys())),
                np.mean(list(val_nmi.values())),
                np.std(list(val_nmi.values()), ddof=1) / np.sqrt(len(val_nmi.keys())),
                np.mean(list(test_rand.values())),
                np.std(list(test_rand.values()), ddof=1) / np.sqrt(len(test_rand.keys())),
                np.mean(list(test_nmi.values())),
                np.std(list(test_nmi.values()), ddof=1) / np.sqrt(len(test_nmi.keys()))
    ))


def treccar_clustering_single_model_with_sections(treccar_2cv_data_file,
                                    device,
                                    clustering_loss_name,
                                    query_context_ref,
                                    max_num_tokens,
                                    max_grad_norm,
                                    weight_decay,
                                    warmup,
                                    lrate,
                                    num_epochs,
                                    emb_model_name,
                                    emb_dim,
                                    output_path):
    if query_context_ref is not None:
        with open(query_context_ref, 'r') as f:
            qc = json.load(f)
    cv_datasets = np.load(treccar_2cv_data_file, allow_pickle=True)[()]['data']
    alpha = 0.005
    beta = 0.005
    for i in range(len(cv_datasets)):
        train_data_current = cv_datasets[i]
        test_data_current = train_data_current.test_samples
        num_steps_per_epoch = len(train_data_current)
        num_train_steps = num_epochs * num_steps_per_epoch
        model = QuerySpecificClusteringModelWithSection(emb_model_name, emb_dim, device, max_num_tokens)
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
        if clustering_loss_name == 'nmi':
            cl_loss_func = get_nmi_loss
        else:
            cl_loss_func = get_weighted_adj_rand_loss
        for epoch in tqdm(range(num_epochs)):
            running_loss = 0
            for idx in tqdm(range(len(train_data_current))):
                model.train()
                sample = train_data_current[idx]
                query_content = sample.q.split('enwiki:')[1].replace('%20', ' ')
                n = len(sample.paras)
                k = len(set(sample.para_labels))
                input_texts = [(query_content, t) for t in sample.para_texts]
                input_features = model.qp_model.tokenize(input_texts)
                put_features_in_device(input_features, device)
                section_texts = []
                for s in sample.para_labels:
                    if '/' in s:
                        section_texts.append((query_content, s.split('/')[1].replace('%20', ' ')))
                    else:
                        section_texts.append((query_content, query_content))
                section_features = model.qp_model.tokenize(section_texts)
                put_features_in_device(section_features, device)
                #print(GPUtil.showUtilization())
                mc, ma, qp_emb, qs_emb = model(input_features, section_features, k)
                clustering_loss = cl_loss_func(ma, sample.para_labels, device)
                classification_loss = torch.sum((qp_emb - qs_emb).pow(2).sum(1).sqrt())
                section_emb_reg = torch.sum(torch.cdist(qs_emb, qs_emb, p=2.0)) / k ** 2
                loss = clustering_loss + alpha * classification_loss - beta * section_emb_reg
                loss.backward()
                running_loss += loss.item()
                #print(batch.q + ' %d paras, Loss %.4f' % (len(batch.paras), loss.detach().item()))
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
                opt.zero_grad()
                schd.step()
            if query_context_ref is not None:
                test_rand, test_nmi = do_eval(test_data_current, model, qc)
            else:
                test_rand, test_nmi = do_eval(test_data_current, model)
            print('Epoch %d, mean loss: %.4f, mean RAND %.4f +- %.4f, mean NMI %.4f +- %.4f' % (epoch + 1,
                                                                                                running_loss / len(train_data_current),
                                                                                                np.mean(list(test_rand.values())),
                                                                                                np.std(list(test_rand.values()), ddof=1) / np.sqrt(len(test_rand.keys())),
                                                                                                np.mean(list(test_nmi.values())),
                                                                                                np.std(list(test_nmi.values()), ddof=1) / np.sqrt(len(test_nmi.keys()))))
        if output_path is not None:
            print('Saving the trained model...')
            torch.save(model.state_dict(), output_path+'_fold'+str(i+1)+'.model')
            model = QuerySpecificClusteringModel(emb_model_name, emb_dim, device, max_num_tokens)
            model.load_state_dict(torch.load(output_path+'_fold'+str(i+1)+'.model'))
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


def treccar_clustering_baseline_sbert_triplet_model(treccar_2cv_data_file,
                                    device,
                                    max_num_tokens,
                                    max_grad_norm,
                                    weight_decay,
                                    warmup,
                                    lrate,
                                    num_epochs,
                                    emb_model_name,
                                    emb_dim,
                                    output_path,
                                    triplet_margin=5,
                                    batch_size=32):
    cv_datasets = np.load(treccar_2cv_data_file, allow_pickle=True)[()]['data']
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
            running_loss = 0
            n = 0
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
                    running_loss += loss.item()
                    n += 1
            test_rand, test_nmi = do_eval(test_data_current, model, triplet_model=True)
            print('Epoch %d, mean loss: %.4f, mean RAND %.4f +- %.4f, mean NMI %.4f +- %.4f' % (epoch + 1,
                                                                                                running_loss / n,
                                                                                                np.mean(list(test_rand.values())),
                                                                                                np.std(list(test_rand.values()), ddof=1) / np.sqrt(len(test_rand.keys())),
                                                                                                np.mean(list(test_nmi.values())),
                                                                                                np.std(list(test_nmi.values()), ddof=1) / np.sqrt(len(test_nmi.keys()))))
        if output_path is not None:
            print('Saving the trained model...')
            torch.save(model.state_dict(), output_path+'_fold'+str(i+1)+'.model')
            model = SBERTTripletLossModel(emb_model_name, device, max_num_tokens, triplet_margin)
            model.load_state_dict(torch.load(output_path+'_fold'+str(i+1)+'.model'))
        print('Evaluation Fold %d' % (i+1))
        print('=================')
        test_rand, test_nmi = do_eval(test_data_current, model, triplet_model=True)
        print('Mean RAND %.4f +- %.4f, NMI %.4f +- %.4f' % (np.mean(list(test_rand.values())),
                                                            np.std(list(test_rand.values()), ddof=1) / np.sqrt(len(test_rand.keys())),
                                                            np.mean(list(test_nmi.values())),
                                                            np.std(list(test_nmi.values()), ddof=1) / np.sqrt(len(test_nmi.keys()))))


def treccar_clustering_baseline_sbert_triplet_model_full_train(treccar_full_data_file,
                                    device,
                                    max_num_tokens,
                                    max_grad_norm,
                                    weight_decay,
                                    warmup,
                                    lrate,
                                    num_epochs,
                                    emb_model_name,
                                    emb_dim,
                                    output_path,
                                    triplet_margin=5,
                                    batch_size=32):
    treccar_dataset = np.load(treccar_full_data_file, allow_pickle=True)[()]['data']
    val_samples = treccar_dataset.val_samples
    test_samples = treccar_dataset.test_samples
    num_steps_per_epoch = len(treccar_dataset)
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
    val_rand, val_nmi = do_eval(val_samples, model, triplet_model=True)
    test_rand, test_nmi = do_eval(test_samples, model, triplet_model=True)
    print('\nInitial evaluation')
    print('Mean Val RAND %.4f +- %.4f, Val NMI %.4f +- %.4f Test RAND %.4f +- %.4f, Test NMI %.4f +- %.4f' % (
        np.mean(list(val_rand.values())),
        np.std(list(val_rand.values()), ddof=1) / np.sqrt(len(val_rand.keys())),
        np.mean(list(val_nmi.values())),
        np.std(list(val_nmi.values()), ddof=1) / np.sqrt(len(val_nmi.keys())),
        np.mean(list(test_rand.values())),
        np.std(list(test_rand.values()), ddof=1) / np.sqrt(len(test_rand.keys())),
        np.mean(list(test_nmi.values())),
        np.std(list(test_nmi.values()), ddof=1) / np.sqrt(len(test_nmi.keys()))
    ))
    for epoch in tqdm(range(num_epochs)):
        for idx in tqdm(range(len(treccar_dataset))):
            model.train()
            sample = treccar_dataset[idx]
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
            if (idx + 1) % 10000 == 0:
                val_rand, val_nmi = do_eval(val_samples, model, triplet_model=True)
                test_rand, test_nmi = do_eval(test_samples, model, triplet_model=True)
                print('Mean Val RAND %.4f +- %.4f, Val NMI %.4f +- %.4f Test RAND %.4f +- %.4f, Test NMI %.4f +- %.4f' % (
                        np.mean(list(val_rand.values())),
                        np.std(list(val_rand.values()), ddof=1) / np.sqrt(len(val_rand.keys())),
                        np.mean(list(val_nmi.values())),
                        np.std(list(val_nmi.values()), ddof=1) / np.sqrt(len(val_nmi.keys())),
                        np.mean(list(test_rand.values())),
                        np.std(list(test_rand.values()), ddof=1) / np.sqrt(len(test_rand.keys())),
                        np.mean(list(test_nmi.values())),
                        np.std(list(test_nmi.values()), ddof=1) / np.sqrt(len(test_nmi.keys()))
                ))
    if output_path is not None:
        print('Saving the trained model...')
        torch.save(model.state_dict(), output_path + '.model')
        model = SBERTTripletLossModel(emb_model_name, device, max_num_tokens, triplet_margin)
        model.load_state_dict(torch.load(output_path + '.model'))
    val_rand, val_nmi = do_eval(val_samples, model, triplet_model=True)
    test_rand, test_nmi = do_eval(test_samples, model, triplet_model=True)
    print('Final evaluation')
    print('================')
    print('Mean Val RAND %.4f +- %.4f, Val NMI %.4f +- %.4f Test RAND %.4f +- %.4f, Test NMI %.4f +- %.4f' % (
        np.mean(list(val_rand.values())),
        np.std(list(val_rand.values()), ddof=1) / np.sqrt(len(val_rand.keys())),
        np.mean(list(val_nmi.values())),
        np.std(list(val_nmi.values()), ddof=1) / np.sqrt(len(val_nmi.keys())),
        np.mean(list(test_rand.values())),
        np.std(list(test_rand.values()), ddof=1) / np.sqrt(len(test_rand.keys())),
        np.mean(list(test_nmi.values())),
        np.std(list(test_nmi.values()), ddof=1) / np.sqrt(len(test_nmi.keys()))
    ))


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


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA is available and using device: '+str(device))
    else:
        device = torch.device('cpu')
        print('CUDA not available, using device: '+str(device))
    parser = argparse.ArgumentParser(description='Run query specific clustering experiments on TRECCAR 2-fold cv dataset')
    parser.add_argument('-td', '--treccar_data', help='Path to TRECCAR clustering npy file prepared for 2-fold cv',
                        default='D:\\new_cats_data\\QSC_data\\benchmarkY1-train-nodup\\treccar_clustering_data_by1train_2cv.npy')
    parser.add_argument('-op', '--output_path', help='Path to save the trained model', default=None)
    parser.add_argument('-ne', '--experiment', type=int, help='Choose the experiment to run (1: QSC/ 2: QSC treccar full/ 3: QSC section/ 4: baseline/ 5: baseline full)', default=2)
    parser.add_argument('-ls', '--loss', help='Loss func to use for QSC', default='adj')
    parser.add_argument('-qc', '--query_con', help='Path to query-context json file', default=None)
    parser.add_argument('-mn', '--model_name', help='SBERT embedding model name', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('-nt', '--max_num_tokens', type=int, help='Max no. of tokens', default=128)
    parser.add_argument('-gn', '--max_grad_norm', type=float, help='Max gradient norm', default=1.0)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.01)
    parser.add_argument('-wu', '--warmup', type=int, default=1000)
    parser.add_argument('-lr', '--lrate', type=float, default=2e-5)
    parser.add_argument('-ep', '--epochs', type=int, default=75)
    parser.add_argument('-ed', '--emb_dim', type=int, default=256)
    parser.add_argument('--nq', action='store_true', default=False)

    args = parser.parse_args()
    if args.experiment == 1:
        treccar_clustering_single_model(args.treccar_data, device, args.loss, args.query_con, args.max_num_tokens,
                                        args.max_grad_norm, args.weight_decay, args.warmup, args.lrate, args.epochs,
                                        args.model_name, args.emb_dim, args.output_path)
    elif args.experiment == 2:
        treccar_clustering_single_model_full_train(args.treccar_data, args.nq, device, args.loss, args.query_con, args.max_num_tokens,
                                        args.max_grad_norm, args.weight_decay, args.warmup, args.lrate, args.epochs,
                                        args.model_name, args.emb_dim, args.output_path)
    elif args.experiment == 3:
        treccar_clustering_single_model_with_sections(args.treccar_data, device, args.loss, args.query_con, args.max_num_tokens,
                                        args.max_grad_norm, args.weight_decay, args.warmup, args.lrate, args.epochs,
                                        args.model_name, args.emb_dim, args.output_path)
    elif args.experiment == 4:
        treccar_clustering_baseline_sbert_triplet_model(args.treccar_data, device, args.max_num_tokens, args.max_grad_norm,
                                                        args.weight_decay, args.warmup, args.lrate, args.epochs,
                                                        args.model_name, args.emb_dim, args.output_path)
    elif args.experiment == 5:
        treccar_clustering_baseline_sbert_triplet_model_full_train(args.treccar_data, device, args.max_num_tokens, args.max_grad_norm,
                                                        args.weight_decay, args.warmup, args.lrate, args.epochs,
                                                        args.model_name, args.emb_dim, args.output_path)


if __name__ == '__main__':
    main()