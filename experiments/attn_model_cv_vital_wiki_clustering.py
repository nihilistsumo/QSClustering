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
    get_weighted_adj_rand_loss, QuerySpecificDKM, QuerySpecificClusteringModelWithSection, QuerySpecificAttentionClusteringModel
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
            query_content = s.category
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


def vital_wiki_clustering_attn_model(vital_wiki_2cv_data_file,
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
                                    attn_emb_dim,
                                    num_attn_head,
                                    kmeans_plus,
                                    output_path):
    if query_context_ref is not None:
        with open(query_context_ref, 'r') as f:
            qc = json.load(f)
    cv_datasets = np.load(vital_wiki_2cv_data_file, allow_pickle=True)[()]['data']
    for i in range(len(cv_datasets)):
        train_data_current = cv_datasets[i]
        test_data_current = train_data_current.test_samples
        num_steps_per_epoch = len(train_data_current)
        num_train_steps = num_epochs * num_steps_per_epoch
        model = QuerySpecificAttentionClusteringModel(emb_model_name, attn_emb_dim, num_attn_head, device,
                                                      max_num_tokens, kmeans_plus)
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
            test_rand, test_nmi = do_eval(test_data_current, model, qc, no_query=no_query_mode)
        else:
            test_rand, test_nmi = do_eval(test_data_current, model, no_query=no_query_mode)
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
                query_content = sample.category +'. ' + sample.q.split('enwiki:')[1].replace('%20', ' ')
                n = len(sample.paras)
                k = len(set(sample.para_labels))
                if no_query_mode:
                    input_texts = sample.para_texts
                else:
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
                test_rand, test_nmi = do_eval(test_data_current, model, qc, no_query=no_query_mode)
            else:
                test_rand, test_nmi = do_eval(test_data_current, model, no_query=no_query_mode)
            print('Epoch %d, mean loss: %.4f, mean RAND %.4f +- %.4f, mean NMI %.4f +- %.4f' % (epoch + 1,
                                                                running_loss / len(train_data_current),
                                                                np.mean(list(test_rand.values())),
                                                                np.std(list(test_rand.values()), ddof=1) / np.sqrt(len(test_rand.keys())),
                                                                np.mean(list(test_nmi.values())),
                                                                np.std(list(test_nmi.values()), ddof=1) / np.sqrt(len(test_nmi.keys()))))
        if output_path is not None:
            print('Saving the trained model...')
            torch.save(model.state_dict(), output_path+'_fold'+str(i+1)+'.model')
            model = QuerySpecificClusteringModel(emb_model_name, attn_emb_dim, device, max_num_tokens)
            model.load_state_dict(torch.load(output_path+'_fold'+str(i+1)+'.model'))
        print('Evaluation Fold %d' % (i+1))
        print('=================')
        if query_context_ref is not None:
            test_rand, test_nmi = do_eval(test_data_current, model, qc, no_query=no_query_mode)
        else:
            test_rand, test_nmi = do_eval(test_data_current, model, no_query=no_query_mode)
        print('Mean RAND %.4f +- %.4f, NMI %.4f +- %.4f' % (np.mean(list(test_rand.values())),
                                                            np.std(list(test_rand.values()), ddof=1) / np.sqrt(len(test_rand.keys())),
                                                            np.mean(list(test_nmi.values())),
                                                            np.std(list(test_nmi.values()), ddof=1) / np.sqrt(len(test_nmi.keys()))))


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA is available and using device: '+str(device))
    else:
        device = torch.device('cpu')
        print('CUDA not available, using device: '+str(device))
    parser = argparse.ArgumentParser(description='Run query specific clustering experiments on Vital wiki 2-fold cv dataset')
    parser.add_argument('-vd', '--vital_data', help='Path to vital wiki clustering npy file prepared for 2-fold cv',
                        default='D:\\new_cats_data\\QSC_data\\vital_wiki\\vital_wiki_clustering_data_2cv.npy')
    parser.add_argument('-op', '--output_path', help='Path to save the trained model', default=None)
    parser.add_argument('-ne', '--experiment', type=int, help='Choose the experiment to run (1: QSC/ 2: QSC section/ 3: DKM param/ 4: baseline)', default=1)
    parser.add_argument('-ls', '--loss', help='Loss func to use for QSC', default='adj')

    parser.add_argument('-qc', '--query_con', help='Path to query-context json file', default=None)
    parser.add_argument('-mn', '--model_name', help='SBERT embedding model name',
                        default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('-nt', '--max_num_tokens', help='Max no. of tokens', type=int, default=128)
    parser.add_argument('-gn', '--max_grad_norm', help='Max gradient norm', type=float, default=1.0)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.01)
    parser.add_argument('-wu', '--warmup', type=int, default=1000)
    parser.add_argument('-lr', '--lrate', type=float, default=2e-5)
    parser.add_argument('-ep', '--epochs', type=int, default=2)
    parser.add_argument('-ed', '--attn_emb_dim', type=int, default=128)
    parser.add_argument('-na', '--num_attn_head', type=int, default=4)
    parser.add_argument('--nq', action='store_true', default=False)
    parser.add_argument('--kp', action='store_true', default=False)

    args = parser.parse_args()
    if args.experiment == 1:
        vital_wiki_clustering_attn_model(args.vital_data, args.nq, device, args.loss, args.query_con, args.max_num_tokens,
                                           args.max_grad_norm, args.weight_decay, args.warmup, args.lrate, args.epochs,
                                           args.model_name, args.attn_emb_dim, args.num_attn_head, args.kp, args.output_path)


if __name__ == '__main__':
    main()