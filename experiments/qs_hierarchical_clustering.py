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
from sklearn.cluster import AgglomerativeClustering
from core.clustering import QS3M_HACModel, QuerySpecificHACModel
from experiments.cv_treccar_clustering import do_eval, do_eval_qs3m, put_features_in_device
from tqdm import tqdm, trange
import json
import numpy as np
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


def true_adj_mat(label):
    n = len(label)
    adj_mat = torch.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j or label[i] == label[j]:
                adj_mat[i][j] = 1.0
    return adj_mat


def clustering(batch_pairscore_matrix, num_clusters):
    batch_adjacency_matrix = np.zeros(batch_pairscore_matrix.shape)
    cl = AgglomerativeClustering(n_clusters=num_clusters, affinity='precomputed', linkage='average')
    cluster_label = cl.fit_predict(batch_pairscore_matrix)
    for m in range(cluster_label.shape[0]):
        for n in range(cluster_label.shape[0]):
            if cluster_label[m] == cluster_label[n]:
                batch_adjacency_matrix[m][n] = 1.0
    return batch_adjacency_matrix, cluster_label


class OptimCluster(torch.autograd.Function):

    @staticmethod
    def forward(ctx, batch_pairscore_matrix, lambda_val, num_clusters):
        ctx.lambda_val = lambda_val
        ctx.num_clusters = num_clusters
        ctx.batch_pairscore_matrix = batch_pairscore_matrix.detach().cpu().numpy()
        ctx.batch_adj_matrix, _ = clustering(ctx.batch_pairscore_matrix, ctx.num_clusters)
        return torch.from_numpy(ctx.batch_adj_matrix).float().to(batch_pairscore_matrix.device)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output_numpy = grad_output.detach().cpu().numpy()
        batch_pairscore_matrix_prime = np.maximum(ctx.batch_pairscore_matrix + ctx.lambda_val * grad_output_numpy, 0.0)
        better_batch_adj_matrix, _ = clustering(batch_pairscore_matrix_prime, ctx.num_clusters)
        gradient = -(ctx.batch_adj_matrix - better_batch_adj_matrix) / ctx.lambda_val
        return torch.from_numpy(gradient.astype(np.float32)).to(grad_output.device), None, None


def treccar_clustering_hac_cob_full_train(treccar_full_data_file,
                                    no_query_mode,
                                    device,
                                    lambda_val,
                                    reg_val,
                                    query_context_ref,
                                    max_num_tokens,
                                    max_grad_norm,
                                    weight_decay,
                                    warmup,
                                    lrate,
                                    num_epochs,
                                    emb_model_name,
                                    emb_dim,
                                    val_step,
                                    output_path):
    if query_context_ref is not None:
        with open(query_context_ref, 'r') as f:
            qc = json.load(f)
    treccar_dataset = np.load(treccar_full_data_file, allow_pickle=True)[()]['data']
    val_samples = treccar_dataset.val_samples
    test_samples = treccar_dataset.test_samples
    num_steps_per_epoch = len(treccar_dataset)
    num_train_steps = num_epochs * num_steps_per_epoch
    model = QuerySpecificHACModel(emb_model_name, emb_dim, device, max_num_tokens)
    model_params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model_params if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    opt = AdamW(optimizer_grouped_parameters, lr=lrate)
    bb_optim = OptimCluster()
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
    for epoch in range(num_epochs):
        print('Epoch %d\n=========' % (epoch + 1))
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
            true_adjacency_mat = true_adj_mat(sample.para_labels).to(device)
            dist_mat = model(input_features)
            mean_similar_dist = (dist_mat * true_adjacency_mat).sum() / true_adjacency_mat.sum()
            mean_dissimilar_dist = (dist_mat * (1.0 - true_adjacency_mat)).sum() / (1 - true_adjacency_mat).sum()
            adjacency_mat = bb_optim.apply(dist_mat, lambda_val, k).to(device)
            weighted_err_mat = adjacency_mat * (1.0 - true_adjacency_mat) + (1.0 - adjacency_mat) * true_adjacency_mat
            weighted_err_mean = weighted_err_mat.mean(dim=0).sum()
            loss = weighted_err_mean + reg_val * (mean_similar_dist - mean_dissimilar_dist)
            loss.backward()
            # print(batch.q + ' %d paras, Loss %.4f' % (len(batch.paras), loss.detach().item()))
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            opt.zero_grad()
            schd.step()
            if (idx + 1) % val_step == 0:
                if query_context_ref is not None:
                    val_rand, val_nmi = do_eval(val_samples, model, qc)
                    test_rand, test_nmi = do_eval(test_samples, model, qc)
                else:
                    val_rand, val_nmi = do_eval(val_samples, model, no_query=no_query_mode)
                    test_rand, test_nmi = do_eval(test_samples, model, no_query=no_query_mode)
                print('\nMean Val RAND %.4f +- %.4f, Val NMI %.4f +- %.4f Test RAND %.4f +- %.4f, Test NMI %.4f +- %.4f' % (
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
        model = QuerySpecificHACModel(emb_model_name, emb_dim, device, max_num_tokens)
        model.load_state_dict(torch.load(output_path + '.model'))
    if query_context_ref is not None:
        val_rand, val_nmi = do_eval(val_samples, model, qc)
        test_rand, test_nmi = do_eval(test_samples, model, qc)
    else:
        val_rand, val_nmi = do_eval(val_samples, model, no_query=no_query_mode)
        test_rand, test_nmi = do_eval(test_samples, model, no_query=no_query_mode)
    print('\nFinal Evaluation')
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


def treccar_clustering_qs3m_full_train(treccar_full_data_file,
                                    device,
                                    query_context_ref,
                                    max_num_tokens,
                                    max_grad_norm,
                                    weight_decay,
                                    warmup,
                                    lrate,
                                    num_epochs,
                                    emb_model_name,
                                    emb_dim,
                                    val_step,
                                    output_path):
    if query_context_ref is not None:
        with open(query_context_ref, 'r') as f:
            qc = json.load(f)
    treccar_dataset = np.load(treccar_full_data_file, allow_pickle=True)[()]['data']
    val_samples = treccar_dataset.val_samples
    test_samples = treccar_dataset.test_samples
    num_steps_per_epoch = len(treccar_dataset)
    num_train_steps = num_epochs * num_steps_per_epoch
    model = QS3M_HACModel(emb_model_name, emb_dim, device, max_num_tokens).to(device)
    model_params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model_params if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    opt = AdamW(optimizer_grouped_parameters, lr=lrate)
    mse_loss = nn.MSELoss()
    schd = transformers.get_linear_schedule_with_warmup(opt, warmup, num_epochs * num_train_steps)
    model.eval()
    if query_context_ref is not None:
        val_rand, val_nmi = do_eval_qs3m(val_samples, model, qc)
        test_rand, test_nmi = do_eval_qs3m(test_samples, model, qc)
    else:
        val_rand, val_nmi = do_eval_qs3m(val_samples, model)
        test_rand, test_nmi = do_eval_qs3m(test_samples, model)
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
    for epoch in range(num_epochs):
        print('Epoch %d\n=========' % (epoch + 1))
        for idx in tqdm(range(len(treccar_dataset))):
            model.train()
            sample = treccar_dataset[idx]
            query_content = sample.q.split('enwiki:')[1].replace('%20', ' ')
            n = len(sample.paras)
            k = len(set(sample.para_labels))
            input_texts = sample.para_texts
            # print(GPUtil.showUtilization())
            true_adjacency_mat = true_adj_mat(sample.para_labels).to(device)
            sim_mat = model(query_content, input_texts)
            loss = mse_loss(sim_mat, true_adjacency_mat)
            loss.backward()
            # print(batch.q + ' %d paras, Loss %.4f' % (len(batch.paras), loss.detach().item()))
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            opt.zero_grad()
            schd.step()
            if (idx + 1) % val_step == 0:
                model.eval()
                if query_context_ref is not None:
                    val_rand, val_nmi = do_eval_qs3m(val_samples, model, qc)
                    test_rand, test_nmi = do_eval_qs3m(test_samples, model, qc)
                else:
                    val_rand, val_nmi = do_eval_qs3m(val_samples, model)
                    test_rand, test_nmi = do_eval_qs3m(test_samples, model)
                print('\nMean Val RAND %.4f +- %.4f, Val NMI %.4f +- %.4f Test RAND %.4f +- %.4f, Test NMI %.4f +- %.4f' % (
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
        model = QS3M_HACModel(emb_model_name, emb_dim, device, max_num_tokens).to(device)
        model.load_state_dict(torch.load(output_path + '.model'))
    model.eval()
    if query_context_ref is not None:
        val_rand, val_nmi = do_eval_qs3m(val_samples, model, qc)
        test_rand, test_nmi = do_eval_qs3m(test_samples, model, qc)
    else:
        val_rand, val_nmi = do_eval_qs3m(val_samples, model)
        test_rand, test_nmi = do_eval_qs3m(test_samples, model)
    print('\nFinal Evaluation')
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
    parser = argparse.ArgumentParser(description='Run query specific clustering experiments on TRECCAR 2-fold cv dataset')
    parser.add_argument('-td', '--treccar_data', help='Path to TRECCAR clustering npy file prepared for 2-fold cv',
                        default='D:\\new_cats_data\\QSC_data\\train\\treccar_train_clustering_data_full.npy')
    parser.add_argument('-op', '--output_path', help='Path to save the trained model', default=None)
    parser.add_argument('-ne', '--experiment', type=int, help='Choose the experiment to run', default=1)
    parser.add_argument('-qc', '--query_con', help='Path to query-context json file', default=None)
    parser.add_argument('-mn', '--model_name', help='SBERT embedding model name', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('-nt', '--max_num_tokens', type=int, help='Max no. of tokens', default=128)
    parser.add_argument('-gn', '--max_grad_norm', type=float, help='Max gradient norm', default=1.0)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.01)
    parser.add_argument('-wu', '--warmup', type=int, default=1000)
    parser.add_argument('-lr', '--lrate', type=float, default=2e-5)
    parser.add_argument('-lm', '--lambda_val', type=float, default=200.0)
    parser.add_argument('-rg', '--reg_val', type=float, default=2.5)
    parser.add_argument('-ep', '--epochs', type=int, default=2)
    parser.add_argument('-vs', '--val_step', type=int, default=10000)
    parser.add_argument('-ed', '--emb_dim', type=int, default=256)
    parser.add_argument('--nq', action='store_true', default=False)

    args = parser.parse_args()
    if args.experiment == 1:
        treccar_clustering_hac_cob_full_train(args.treccar_data, args.nq, device, args.lambda_val, args.reg_val,
                                               args.query_con, args.max_num_tokens, args.max_grad_norm,
                                               args.weight_decay, args.warmup, args.lrate, args.epochs, args.model_name,
                                               args.emb_dim, args.val_step, args.output_path)
    elif args.experiment == 2:
        treccar_clustering_qs3m_full_train(args.treccar_data, device, args.query_con, args.max_num_tokens,
                                           args.max_grad_norm, args.weight_decay, args.warmup, args.lrate, args.epochs,
                                           args.model_name, args.emb_dim, args.val_step, args.output_path)


if __name__ == '__main__':
    main()