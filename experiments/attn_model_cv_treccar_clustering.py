import math
import random
import argparse
import transformers
from transformers import AdamW
from sentence_transformers import SentenceTransformer
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from core.clustering import QuerySpecificClusteringModel, SBERTTripletLossModel, get_nmi_loss, \
    get_weighted_adj_rand_loss, QuerySpecificDKM, QuerySpecificClusteringModelWithSection, \
    QuerySpecificAttentionClusteringModel, QuerySpecificAttentionFixedEmbedClusteringModel
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


def do_eval(test_samples, model, triplet_model=False):
    model.eval()
    rand_dict, nmi_dict = {}, {}
    for s in test_samples:
        true_labels = s.para_labels
        k = len(set(true_labels))
        if triplet_model:
            embeddings = model.emb_model.encode(s.para_texts, convert_to_tensor=True)
        else:
            embeddings = model.get_embedding_without_attn(s.para_texts)
        pred_labels = model.get_clustering(embeddings, k)
        rand = adjusted_rand_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        rand_dict[s.q] = rand
        nmi_dict[s.q] = nmi
    return rand_dict, nmi_dict


def do_attn_fixed_emb_eval(test_samples, test_psg_embeddings, model, device):
    model.eval()
    rand_dict, nmi_dict = {}, {}
    for s in test_samples:
        true_labels = s.para_labels
        k = len(set(true_labels))
        embeddings = test_psg_embeddings[s.q].to(device)
        pred_labels = model.get_clustering(embeddings, k)
        rand = adjusted_rand_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        rand_dict[s.q] = rand
        nmi_dict[s.q] = nmi
    return rand_dict, nmi_dict


def treccar_clustering_fixed_emb_attn_model_full_train(treccar_full_data_file,
                                    device,
                                    loss_name,
                                    max_grad_norm,
                                    weight_decay,
                                    warmup,
                                    lrate,
                                    num_epochs,
                                    emb_model,
                                    attn_dim,
                                    num_attn_head,
                                    kmeans_plus,
                                    output_path):
    treccar_dataset = np.load(treccar_full_data_file, allow_pickle=True)[()]['data']
    train_samples = [treccar_dataset[i] for i in range(len(treccar_dataset))]
    val_samples = treccar_dataset.val_samples
    test_samples = treccar_dataset.test_samples
    treccar_clustering_fixed_emb_attn_model(train_samples, val_samples, test_samples, device, loss_name, max_grad_norm,
                                            weight_decay, warmup, lrate, num_epochs, emb_model, attn_dim,
                                            num_attn_head, kmeans_plus, output_path)


def treccar_clustering_fixed_emb_attn_model_existing_emb(treccar_full_data_file,
                                    device,
                                    loss_name,
                                    max_grad_norm,
                                    weight_decay,
                                    warmup,
                                    lrate,
                                    num_epochs,
                                    trained_qs_model,
                                    attn_dim,
                                    num_attn_head,
                                    kmeans_plus,
                                    output_path):
    treccar_dataset = np.load(treccar_full_data_file, allow_pickle=True)[()]['data']
    #train_samples = treccar_dataset.val_samples
    train_samples = [treccar_dataset[i] for i in range(len(treccar_dataset))]
    val_samples = treccar_dataset.val_samples
    test_samples = treccar_dataset.test_samples
    val_rand, val_nmi = do_eval(val_samples, trained_qs_model)
    test_rand, test_nmi = do_eval(test_samples, trained_qs_model)
    print('\nInitial evaluation without attention')
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
    emb_model = trained_qs_model.qp_model
    treccar_clustering_fixed_emb_attn_model(train_samples, val_samples, test_samples, device, loss_name, max_grad_norm,
                                            weight_decay, warmup, lrate, num_epochs, emb_model, attn_dim,
                                            num_attn_head, kmeans_plus, output_path)


def treccar_clustering_fixed_emb_attn_model(train_samples,
                                    val_samples,
                                    test_samples,
                                    device,
                                    loss_name,
                                    max_grad_norm,
                                    weight_decay,
                                    warmup,
                                    lrate,
                                    num_epochs,
                                    emb_model,
                                    attn_dim,
                                    num_attn_head,
                                    kmeans_plus,
                                    output_path):
    num_steps_per_epoch = len(train_samples)
    num_train_steps = num_epochs * num_steps_per_epoch
    emb_dim = emb_model.get_sentence_embedding_dimension()
    val_psg_embs, test_psg_embs = {}, {}
    for s in val_samples:
        val_psg_embs[s.q] = emb_model.encode(s.para_texts, convert_to_tensor=True)
    for s in test_samples:
        test_psg_embs[s.q] = emb_model.encode(s.para_texts, convert_to_tensor=True)
    model = QuerySpecificAttentionFixedEmbedClusteringModel(emb_dim, attn_dim, num_attn_head, device, kmeans_plus).to(device)
    model_params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model_params if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    opt = AdamW(optimizer_grouped_parameters, lr=lrate)
    schd = transformers.get_linear_schedule_with_warmup(opt, warmup, num_epochs * num_train_steps)
    val_rand, val_nmi = do_attn_fixed_emb_eval(val_samples, val_psg_embs, model, device)
    test_rand, test_nmi = do_attn_fixed_emb_eval(test_samples, test_psg_embs, model, device)
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
        for idx in tqdm(range(len(train_samples))):
            model.train()
            sample = train_samples[idx]
            n = len(sample.paras)
            k = len(set(sample.para_labels))
            input_embeddings = emb_model.encode(sample.para_texts, convert_to_tensor=True, device=device)
            mc, ma = model(input_embeddings, k)
            loss = loss_func(ma, sample.para_labels, device)
            loss.backward()
            # print(batch.q + ' %d paras, Loss %.4f' % (len(batch.paras), loss.detach().item()))
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            opt.zero_grad()
            schd.step()
            if (idx + 1) % 1000 == 0:
                val_rand, val_nmi = do_attn_fixed_emb_eval(val_samples, val_psg_embs, model, device)
                test_rand, test_nmi = do_attn_fixed_emb_eval(test_samples, test_psg_embs, model, device)
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
        val_rand, val_nmi = do_attn_fixed_emb_eval(val_samples, val_psg_embs, model, device)
        test_rand, test_nmi = do_attn_fixed_emb_eval(test_samples, test_psg_embs, model, device)
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
        model = QuerySpecificAttentionFixedEmbedClusteringModel(emb_dim, num_attn_head, device, kmeans_plus)
        model.load_state_dict(torch.load(output_path + '.model'))
    val_rand, val_nmi = do_attn_fixed_emb_eval(val_samples, val_psg_embs, model, device)
    test_rand, test_nmi = do_attn_fixed_emb_eval(test_samples, test_psg_embs, model, device)
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


def treccar_clustering_attn_model_full_train(treccar_full_data_file,
                                    device,
                                    loss_name,
                                    max_num_tokens,
                                    max_grad_norm,
                                    weight_decay,
                                    warmup,
                                    lrate,
                                    num_epochs,
                                    emb_model_path,
                                    emb_model_name,
                                    emb_dim,
                                    attn_dim,
                                    num_attn_head,
                                    kmeans_plus,
                                    output_path):
    treccar_dataset = np.load(treccar_full_data_file, allow_pickle=True)[()]['data']
    train_samples = [treccar_dataset[i] for i in range(len(treccar_dataset))]
    #train_samples = treccar_dataset.val_samples
    val_samples = treccar_dataset.val_samples
    test_samples = treccar_dataset.test_samples
    num_steps_per_epoch = len(treccar_dataset)
    num_train_steps = num_epochs * num_steps_per_epoch
    if emb_model_path is not None:
        m = QuerySpecificClusteringModel(emb_model_name, None, device, max_num_tokens)
        m.load_state_dict(torch.load(emb_model_path))
        model = QuerySpecificAttentionClusteringModel(m.qp_model, emb_dim, attn_dim, num_attn_head, device,
                                                      max_num_tokens, kmeans_plus).to(device)
    else:
        model = QuerySpecificAttentionClusteringModel(emb_model_name, emb_dim, attn_dim, num_attn_head, device,
                                                  max_num_tokens, kmeans_plus).to(device)
    model_params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model_params if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    opt = AdamW(optimizer_grouped_parameters, lr=lrate)
    schd = transformers.get_linear_schedule_with_warmup(opt, warmup, num_epochs * num_train_steps)
    val_rand, val_nmi = do_eval(val_samples, model)
    test_rand, test_nmi = do_eval(test_samples, model)
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
        for idx in tqdm(range(len(train_samples))):
            model.train()
            sample = train_samples[idx]
            n = len(sample.paras)
            k = len(set(sample.para_labels))
            input_texts = sample.para_texts
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
            if (idx + 1) % 1000 == 0:
                val_rand, val_nmi = do_eval(val_samples, model)
                test_rand, test_nmi = do_eval(test_samples, model)
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
        model = QuerySpecificAttentionClusteringModel(emb_model_name, emb_dim, attn_dim, num_attn_head, device,
                                                  max_num_tokens, kmeans_plus)
        model.load_state_dict(torch.load(output_path + '.model'))
    val_rand, val_nmi = do_eval(val_samples, model)
    test_rand, test_nmi = do_eval(test_samples, model)
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


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA is available and using device: '+str(device))
    else:
        device = torch.device('cpu')
        print('CUDA not available, using device: '+str(device))
    parser = argparse.ArgumentParser(description='Run query specific clustering experiments on TRECCAR')
    parser.add_argument('-td', '--treccar_data', help='Path to TRECCAR clustering npy file',
                        default='D:\\new_cats_data\\QSC_data\\train\\treccar_train_clustering_data_full.npy')
    parser.add_argument('-op', '--output_path', help='Path to save the trained model', default=None)
    parser.add_argument('-ne', '--experiment', type=int, help='Choose the experiment to run (1: QSC attn fix emb/'
                                                              '2: QSC attn fix emb train only by1train/ '
                                                              '3: QSC attn treccar full', default=1)
    parser.add_argument('-ls', '--loss', help='Loss func to use for QSC', default='adj')
    parser.add_argument('-qc', '--query_con', help='Path to query-context json file', default=None)
    parser.add_argument('-mn', '--emb_model_name', help='SBERT embedding model name', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('-ed', '--emb_dim', type=int, help='Embedding dim of the sbert model if we use a NN layer on top', default=None)
    parser.add_argument('-ad', '--attn_emb_dim', type=int, help='Embedding dim of the attention model', default=None)
    parser.add_argument('-mp', '--emb_model_path', help='Path to existing trained QS clustering model', default=None)
    parser.add_argument('-nt', '--max_num_tokens', help='Max no. of tokens', type=int, default=128)
    parser.add_argument('-gn', '--max_grad_norm', help='Max gradient norm', type=float, default=1.0)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.01)
    parser.add_argument('-wu', '--warmup', type=int, default=1000)
    parser.add_argument('-lr', '--lrate', type=float, default=2e-5)
    parser.add_argument('-ep', '--epochs', type=int, default=2)
    parser.add_argument('-na', '--num_attn_head', type=int, default=4)
    parser.add_argument('--nq', action='store_true', default=False)
    parser.add_argument('--kp', action='store_true', default=False)

    args = parser.parse_args()
    if args.experiment == 1:
        if args.emb_model_path is None:
            emb_model = SentenceTransformer(args.emb_model_name)
        else:
            m = QuerySpecificClusteringModel(args.emb_model_name, args.emb_dim, device, args.max_num_tokens)
            m.load_state_dict(torch.load(args.emb_model_path))
            emb_model = m.qp_model
        treccar_clustering_fixed_emb_attn_model_full_train(args.treccar_data, device, args.loss,
                                        args.max_grad_norm, args.weight_decay, args.warmup, args.lrate, args.epochs,
                                        emb_model, args.attn_emb_dim, args.num_attn_head,
                                        args.kp, args.output_path)
    elif args.experiment == 2:
        m = QuerySpecificClusteringModel(args.emb_model_name, args.emb_dim, device, args.max_num_tokens)
        m.load_state_dict(torch.load(args.emb_model_path))
        treccar_clustering_fixed_emb_attn_model_existing_emb(args.treccar_data, device, args.loss, args.max_grad_norm,
                                                             args.weight_decay, args.warmup, args.lrate, args.epochs,
                                                             m, args.attn_emb_dim, args.num_attn_head, args.kp, args.output_path)
    elif args.experiment == 3:
        treccar_clustering_attn_model_full_train(args.treccar_data, device, args.loss, args.max_num_tokens,
                                        args.max_grad_norm, args.weight_decay, args.warmup, args.lrate, args.epochs,
                                        args.emb_model_path, args.emb_model_name, args.emb_dim, args.attn_emb_dim,
                                        args.num_attn_head, args.kp, args.output_path)


if __name__ == '__main__':
    main()