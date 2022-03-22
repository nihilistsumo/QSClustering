import random
import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import AdamW
from sklearn.cluster import kmeans_plusplus
from sentence_transformers import models, SentenceTransformer
import ir_measures
from ir_measures import MAP, Rprec, nDCG
from tqdm import tqdm
from util.data import get_article_qrels, get_page_sec_para_dict, TRECCAR_Datset
from experiments.treccar_clustering import put_features_in_device
from core.clustering import DKM, get_nmi_loss, get_weighted_adj_rand_loss, get_adj_rand_loss
random.seed(42)
torch.manual_seed(42)


def prepare_data(art_qrels, qrels, paratext_tsv):
    page_paras = get_article_qrels(art_qrels)
    page_sec_paras = get_page_sec_para_dict(qrels)
    paratext = {}
    with open(paratext_tsv, 'r', encoding='utf-8') as f:
        for l in f:
            p = l.split('\t')[0]
            paratext[p] = l.split('\t')[1].strip()
    return page_paras, page_sec_paras, paratext


def eval_duo_sbert(duo_sbert, page_paras, page_sec_paras, paratext, qrels):
    duo_sbert.eval()
    with open('temp.val.run', 'w') as f:
        pages = list(page_sec_paras.keys())
        for p in tqdm(range(len(pages))):
            page = pages[p]
            cand_set = page_paras[page]
            for sec in page_sec_paras[page].keys():
                if '/' in sec:
                    query = ' '.join(sec.split('/')[1:]).replace('enwiki:', '').replace('%20', ' ')
                else:
                    query = sec.replace('enwiki:', '').replace('%20', ' ')
                texts_for_pi = []
                for i in range(len(cand_set)):
                    pi = cand_set[i]
                    for j in range(len(cand_set)):
                        if i == j:
                            continue
                        pj = cand_set[j]
                        texts_for_pi.append((query, paratext[pi], paratext[pj]))
                rel_score = np.sum(duo_sbert.get_preds(texts_for_pi).flatten())
                f.write(sec + ' 0 ' + pi + ' 0 ' + str(rel_score) + ' val_runid\n')
    qrels_dat = ir_measures.read_trec_qrels(qrels)
    run_dat = ir_measures.read_trec_run('temp.val.run')
    return ir_measures.calc_aggregate([MAP, Rprec, nDCG], qrels_dat, run_dat)


def eval_clustering_duo_sbert(duo_sbert, page_paras, page_sec_paras, paratext, qrels, device):
    duo_sbert.eval()
    pages = list(page_sec_paras.keys())
    for p in tqdm(range(len(pages))):
        page = pages[p]
        paras = []
        true_labels = []
        for s in page_sec_paras[page].keys():
            for para in page_sec_paras[page][s]:
                paras.append(para)
                true_labels.append(s)
        texts = [('', paratext[pi], '') for pi in paras]




def eval_duo_sbert_emb(duo_sbert_emb, page_paras, page_sec_paras, paratext, qrels):
    duo_sbert_emb.eval()
    with open('temp.val.run', 'w') as f:
        pages = list(page_sec_paras.keys())
        for p in tqdm(range(len(pages))):
            page = pages[p]
            cand_set = page_paras[page]
            for sec in page_sec_paras[page].keys():
                if '/' in sec:
                    query = ' '.join(sec.split('/')[1:]).replace('enwiki:', '').replace('%20', ' ')
                else:
                    query = sec.replace('enwiki:', '').replace('%20', ' ')
                for i in range(len(cand_set)):
                    pi = cand_set[i]
                    texts_for_pi = {'q': [], 'p1': [], 'p2': []}
                    for j in range(len(cand_set)):
                        if i == j:
                            continue
                        pj = cand_set[j]
                        texts_for_pi['q'].append(query)
                        texts_for_pi['p1'].append(paratext[pi])
                        texts_for_pi['p2'].append(paratext[pj])
                    rel_score = np.sum(duo_sbert_emb.get_preds(texts_for_pi['q'], texts_for_pi['p1'], texts_for_pi['p2']))
                    f.write(sec + ' 0 ' + pi + ' 0 ' + str(rel_score) + ' val_runid\n')
    qrels_dat = ir_measures.read_trec_qrels(qrels)
    run_dat = ir_measures.read_trec_run('temp.val.run')
    return ir_measures.calc_aggregate([MAP, Rprec, nDCG], qrels_dat, run_dat)


def is_fit_for_training(paras, sec_paras):
    return 10 <= len(paras) <= 200 and len(sec_paras.keys()) > 2


class Duo_SBERT(nn.Module):
    def __init__(self, sbert_emb_model, device):
        super(Duo_SBERT, self).__init__()
        self.emb_model = sbert_emb_model
        self.fc1 = nn.Linear(in_features=sbert_emb_model.get_sentence_embedding_dimension(), out_features=1).to(device)
        self.act = nn.Sigmoid()

    def forward(self, input_fet):
        output_emb = self.emb_model(input_fet)['sentence_embedding']
        pred_score = self.act(self.fc1(output_emb))
        return pred_score

    def get_preds(self, input_texts, cand_set):
        n = len(cand_set) - 1
        output_emb = self.emb_model.encode(input_texts, convert_to_tensor=True)
        output_emb = output_emb.reshape()
        pred_score = self.act(self.fc1(output_emb)).detach().cpu().numpy().flatten()
        return pred_score


class QSClustering_Model(nn.Module):
    def __init__(self, sbert_emb_model, device, kmeans_plus=False):
        super(QSClustering_Model, self).__init__()
        self.emb_model = sbert_emb_model
        self.device = device
        self.dkm = DKM()
        self.use_kmeans_plus = kmeans_plus

    def forward(self, input_fet, k=2):
        self.qp = self.emb_model(input_fet)['sentence_embedding']
        if self.use_kmeans_plus:
            qp = self.qp.detach().clone().cpu().numpy()
            init_c, _ = kmeans_plusplus(qp, k)
            init_c = torch.tensor(init_c, dtype=torch.float32, device=self.device)
        else:
            init_c = self.qp[random.sample(range(self.qp.shape[0]), k)].detach().clone()
        self.C, self.a = self.dkm(self.qp, init_c)
        return self.C, self.a

    def get_clustering(self, embeddings, k=2, debug_switch=False):
        #self.qp = self.qp_model(input_features)['sentence_embedding']
        if self.use_kmeans_plus:
            embeddings = embeddings.detach().clone().cpu().numpy()
            init_c, _ = kmeans_plusplus(embeddings, k)
            init_c = torch.tensor(init_c, dtype=torch.float32, device=self.device)
        else:
            init_c = embeddings[random.sample(range(embeddings.shape[0]), k)].detach().clone()
        self.C, self.a = self.dkm(embeddings, init_c)
        if debug_switch:
            c_np = self.C.clone().cpu().numpy()
            a_np = self.a.clone().cpu().numpy()
            init_c_np = init_c.clone().cpu().numpy()
            embeddings_np = embeddings.clone().cpu().numpy()
            if torch.std(self.a).item() < 0.01:
                print('Low std in attention matrix')
        pred_labels = torch.argmax(self.a, dim=1).detach().cpu().numpy()
        return pred_labels



def train_duo_sbert(art_qrels,
                    qrels,
                    paratext_tsv,
                    val_art_qrels,
                    val_qrels,
                    val_paratext_tsv,
                    test_art_qrels,
                    test_qrels,
                    test_paratext_tsv,
                    device,
                    model_out,
                    trans_model_name='sentence-transformers/all-MiniLM-L6-v2',
                    max_len=512,
                    max_grad_norm=1.0,
                    weight_decay=0.01,
                    warmup=10000,
                    lrate=2e-5,
                    num_epochs=3,
                    batch_size=8,
                    val_step=1000):
    page_paras, page_sec_paras, paratext = prepare_data(art_qrels, qrels, paratext_tsv)
    val_page_paras, val_page_sec_paras, val_paratext = prepare_data(val_art_qrels, val_qrels, val_paratext_tsv)
    test_page_paras, test_page_sec_paras, test_paratext = prepare_data(test_art_qrels, test_qrels, test_paratext_tsv)
    duo_sbert = Duo_SBERT(trans_model_name, max_len, device)
    model_params = list(duo_sbert.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model_params if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    opt = AdamW(optimizer_grouped_parameters, lr=lrate)
    train_data_len = 0
    for p in page_sec_paras.keys():
        train_data_len += len(page_sec_paras[p].keys())
    schd = transformers.get_linear_schedule_with_warmup(opt, warmup, num_epochs * train_data_len)
    val_eval_score = 0
    train_pages = list(page_sec_paras.keys())
    for p in tqdm(range(len(train_pages)), desc='Training loop'):
        page = train_pages[p]
        if page not in page_paras.keys() or not is_fit_for_training(page_paras[page], page_sec_paras[page]):
            continue
        paras = page_paras[page]
        duo_sbert.train()
        for sec in page_sec_paras[page].keys():
            if '/' in sec:
                query = ' '.join(sec.split('/')[1:]).replace('enwiki:', '').replace('%20', ' ')
            else:
                query = sec.replace('enwiki:', '').replace('%20', ' ')
            rel_paras = page_sec_paras[page][sec]
            nonrel_paras = [p for p in paras if p not in rel_paras]
            random.shuffle(rel_paras)
            random.shuffle(nonrel_paras)
            rel_first_texts, nonrel_first_texts = [], []
            for p1 in rel_paras:
                for p2 in nonrel_paras:
                    rel_first_texts.append((query, paratext[p1], paratext[p2]))
                    nonrel_first_texts.append((query, paratext[p2], paratext[p1]))
            for b in range(len(rel_first_texts) // batch_size):
                rel_first_fet_batch = duo_sbert.emb_model.tokenize(rel_first_texts[b * batch_size: (b+1) * batch_size])
                nonrel_first_fet_batch = duo_sbert.emb_model.tokenize(nonrel_first_texts[b * batch_size: (b+1) * batch_size])
                put_features_in_device(rel_first_fet_batch, device)
                put_features_in_device(nonrel_first_fet_batch, device)
                rel_first_preds = duo_sbert(rel_first_fet_batch)
                nonrel_first_preds = duo_sbert(nonrel_first_fet_batch)
                loss = - torch.sum(torch.log(rel_first_preds)) - torch.sum(torch.log(1 - nonrel_first_preds))
                loss.backward()
                #print('\rLoss: %.4f' % loss.item(), end='')
                nn.utils.clip_grad_norm_(duo_sbert.parameters(), max_grad_norm)
                opt.step()
                opt.zero_grad()
                schd.step()
        if (p+1) % val_step == 0:
            print('\nEvaluating on validation set...')
            val_measures = eval_duo_sbert(duo_sbert, val_page_paras, val_page_sec_paras, val_paratext, val_qrels)
            print('\nValidation eval MAP: %.4f, Rprec: %.4f, nDCG: %.4f' % (val_measures[MAP], val_measures[Rprec],
                                                                      val_measures[nDCG]))
            if val_measures[MAP] > val_eval_score:
                torch.save(duo_sbert, model_out)
                val_eval_score = val_measures[MAP]
    print('\nTraining complete. Evaluating on test set...')
    test_measures = eval_duo_sbert(duo_sbert, test_page_paras, test_page_sec_paras, test_paratext, test_qrels)
    print('\nTest eval MAP: %.4f, Rprec: %.4f, nDCG: %.4f' % (test_measures[MAP], test_measures[Rprec],
                                                                    test_measures[nDCG]))


def train_duo_sbert_with_clustering(art_qrels,
                    qrels,
                    paratext_tsv,
                    val_art_qrels,
                    val_qrels,
                    val_paratext_tsv,
                    test_art_qrels,
                    test_qrels,
                    test_paratext_tsv,
                    device,
                    model_out,
                    trans_model_name='sentence-transformers/all-MiniLM-L6-v2',
                    max_len=512,
                    cluster_data_max_num_paras=35,
                    max_grad_norm=1.0,
                    weight_decay=0.01,
                    warmup=10000,
                    lrate=2e-5,
                    num_epochs=3,
                    batch_size=8,
                    clustering_batch_size=16,
                    switch_step=1000):
    page_paras, page_sec_paras, paratext = prepare_data(art_qrels, qrels, paratext_tsv)
    val_page_paras, val_page_sec_paras, val_paratext = prepare_data(val_art_qrels, val_qrels, val_paratext_tsv)
    test_page_paras, test_page_sec_paras, test_paratext = prepare_data(test_art_qrels, test_qrels, test_paratext_tsv)
    train_cluster_data = TRECCAR_Datset(art_qrels, qrels, paratext_tsv, cluster_data_max_num_paras)
    val_cluster_data = TRECCAR_Datset(val_art_qrels, val_qrels, val_paratext_tsv, cluster_data_max_num_paras)
    test_cluster_data = TRECCAR_Datset(test_art_qrels, test_qrels, test_paratext_tsv, cluster_data_max_num_paras)
    trans_model = models.Transformer(trans_model_name, max_seq_length=max_len)
    pool_model = models.Pooling(trans_model.get_word_embedding_dimension())
    emb_model = SentenceTransformer(modules=[trans_model, pool_model]).to(device)
    duo_sbert = Duo_SBERT(emb_model, device)
    clustering_model = QSClustering_Model(emb_model, device)
    model_params = list(duo_sbert.named_parameters()) + list(clustering_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model_params if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    opt = AdamW(optimizer_grouped_parameters, lr=lrate)
    train_data_len = 0
    for p in page_sec_paras.keys():
        train_data_len += len(page_sec_paras[p].keys())
    schd = transformers.get_linear_schedule_with_warmup(opt, warmup, num_epochs * train_data_len)
    val_eval_score = 0
    train_pages = list(page_sec_paras.keys())
    for i in range((len(train_pages) // switch_step) + 1):
        print('Batch %d/%d' % (i+1, len(train_pages) // switch_step + 1))
        batch_pages = train_pages[i * switch_step: (i + 1) * switch_step]

        for bp in tqdm(range(len(batch_pages))):
            page = batch_pages[bp]
            if page not in page_paras.keys() or not is_fit_for_training(page_paras[page], page_sec_paras[page]):
                continue
            paras = page_paras[page]
            duo_sbert.train()
            for sec in page_sec_paras[page].keys():
                if '/' in sec:
                    query = ' '.join(sec.split('/')[1:]).replace('enwiki:', '').replace('%20', ' ')
                else:
                    query = sec.replace('enwiki:', '').replace('%20', ' ')
                rel_paras = page_sec_paras[page][sec]
                nonrel_paras = [p for p in paras if p not in rel_paras]
                random.shuffle(rel_paras)
                random.shuffle(nonrel_paras)
                rel_first_texts, nonrel_first_texts = [], []
                for p1 in rel_paras:
                    for p2 in nonrel_paras:
                        rel_first_texts.append((query, paratext[p1], paratext[p2]))
                        nonrel_first_texts.append((query, paratext[p2], paratext[p1]))
                for b in range(len(rel_first_texts) // batch_size):
                    rel_first_fet_batch = duo_sbert.emb_model.tokenize(
                        rel_first_texts[b * batch_size: (b + 1) * batch_size])
                    nonrel_first_fet_batch = duo_sbert.emb_model.tokenize(
                        nonrel_first_texts[b * batch_size: (b + 1) * batch_size])
                    put_features_in_device(rel_first_fet_batch, device)
                    put_features_in_device(nonrel_first_fet_batch, device)
                    rel_first_preds = duo_sbert(rel_first_fet_batch)
                    nonrel_first_preds = duo_sbert(nonrel_first_fet_batch)
                    loss = - torch.sum(torch.log(rel_first_preds)) - torch.sum(torch.log(1 - nonrel_first_preds))
                    loss.backward()
                    nn.utils.clip_grad_norm_(duo_sbert.parameters(), max_grad_norm)
                    opt.step()
                    opt.zero_grad()
                    schd.step()
        print('\nEvaluating on validation set...')
        val_measures = eval_duo_sbert(duo_sbert, val_page_paras, val_page_sec_paras, val_paratext, val_qrels)
        print('\nValidation eval MAP: %.4f, Rprec: %.4f, nDCG: %.4f' % (val_measures[MAP], val_measures[Rprec],
                                                                        val_measures[nDCG]))
        if val_measures[MAP] > val_eval_score:
            torch.save(duo_sbert, model_out)
            val_eval_score = val_measures[MAP]

        for bp in tqdm(range(len(batch_pages))):
            page = batch_pages[bp]
            if page not in page_paras.keys() or not is_fit_for_training(page_paras[page], page_sec_paras[page]):
                continue
            paras = page_paras[page]
            random.shuffle(paras)
            para_texts = [paratext[p] for p in paras]
            clustering_model.train()
            for sec in page_sec_paras[page].keys():
                pos_paras = page_sec_paras[page][sec]
                sample_labels = [1 if p in pos_paras else 0 for p in paras]
                sample_labels = sample_labels[:clustering_batch_size]
                if len(set(sample_labels)) < 2:
                    continue
                if '/' in sec:
                    query = ' '.join(sec.split('/')[1:]).replace('enwiki:', '').replace('%20', ' ')
                else:
                    query = sec.replace('enwiki:', '').replace('%20', ' ')
                input_features = clustering_model.emb_model.tokenize([(query, p) for p in para_texts[:clustering_batch_size]])
                put_features_in_device(input_features, device)
                mc, ma = clustering_model(input_features)
                loss = get_weighted_adj_rand_loss(ma, mc, sample_labels, device, False)
                loss.backward()
                # print(batch.q + ' %d paras, Loss %.4f' % (len(batch.paras), loss.detach().item()))
                nn.utils.clip_grad_norm_(clustering_model.parameters(), max_grad_norm)
                opt.step()
                opt.zero_grad()
                schd.step()
        print('\nEvaluating on validation set...')
        val_measures = eval_duo_sbert(duo_sbert, val_page_paras, val_page_sec_paras, val_paratext, val_qrels)
        print('\nValidation eval MAP: %.4f, Rprec: %.4f, nDCG: %.4f' % (val_measures[MAP], val_measures[Rprec],
                                                                        val_measures[nDCG]))
        if val_measures[MAP] > val_eval_score:
            torch.save(duo_sbert, model_out)
            val_eval_score = val_measures[MAP]
    print('\nTraining complete. Evaluating on test set...')
    test_measures = eval_duo_sbert(duo_sbert, test_page_paras, test_page_sec_paras, test_paratext, test_qrels)
    print('\nTest eval MAP: %.4f, Rprec: %.4f, nDCG: %.4f' % (test_measures[MAP], test_measures[Rprec],
                                                              test_measures[nDCG]))




class Duo_Siamese_SBERT(nn.Module):
    def __init__(self, sbert_emb_model_name, query_max_len, psg_max_len, device):
        super(Duo_Siamese_SBERT, self).__init__()
        psg_trans_model = models.Transformer(sbert_emb_model_name, max_seq_length=psg_max_len)
        query_trans_model = models.Transformer(sbert_emb_model_name, max_seq_length=query_max_len)
        pool_model = models.Pooling(psg_trans_model.get_word_embedding_dimension())
        self.psg_sbert = SentenceTransformer(modules=[psg_trans_model, pool_model]).to(device)
        self.query_sbert = SentenceTransformer(modules=[query_trans_model, pool_model]).to(device)
        self.fc1 = nn.Linear(in_features=3 * pool_model.get_sentence_embedding_dimension(), out_features=1).to(device)
        self.act = nn.Sigmoid()

    def forward(self, queries_fet, psg1_fet, psg2_fet):
        queries_emb = self.query_sbert(queries_fet)['sentence_embedding']
        psg1_emb = self.psg_sbert(psg1_fet)['sentence_embedding']
        psg2_emb = self.psg_sbert(psg2_fet)['sentence_embedding']
        concat_emb = torch.hstack((queries_emb, psg1_emb, psg2_emb))
        pred_score = self.act(self.fc1(concat_emb))
        return pred_score

    def get_preds(self, query, psg1, psg2):
        queries_emb = self.query_sbert.encode(query, convert_to_tensor=True)
        psg1_emb = self.psg_sbert.encode(psg1, convert_to_tensor=True)
        psg2_emb = self.psg_sbert.encode(psg2, convert_to_tensor=True)
        concat_emb = torch.hstack((queries_emb, psg1_emb, psg2_emb))
        pred_score = self.act(self.fc1(concat_emb)).detach().cpu().numpy().flatten()
        return pred_score


def train_duo_sbert_emb(art_qrels,
                    qrels,
                    paratext_tsv,
                    val_art_qrels,
                    val_qrels,
                    val_paratext_tsv,
                    test_art_qrels,
                    test_qrels,
                    test_paratext_tsv,
                    device,
                    model_out,
                    trans_model_name='sentence-transformers/all-MiniLM-L6-v2',
                    query_max_len=64,
                    psg_max_len=256,
                    max_grad_norm=1.0,
                    weight_decay=0.01,
                    warmup=10000,
                    lrate=2e-5,
                    num_epochs=3,
                    batch_size=8,
                    val_step=1000):
    page_paras, page_sec_paras, paratext = prepare_data(art_qrels, qrels, paratext_tsv)
    val_page_paras, val_page_sec_paras, val_paratext = prepare_data(val_art_qrels, val_qrels, val_paratext_tsv)
    test_page_paras, test_page_sec_paras, test_paratext = prepare_data(test_art_qrels, test_qrels, test_paratext_tsv)
    duo_sbert_emb = Duo_Siamese_SBERT(trans_model_name, query_max_len, psg_max_len, device)
    model_params = list(duo_sbert_emb.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model_params if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    opt = AdamW(optimizer_grouped_parameters, lr=lrate)
    train_data_len = 0
    for p in page_sec_paras.keys():
        train_data_len += len(page_sec_paras[p].keys())
    schd = transformers.get_linear_schedule_with_warmup(opt, warmup, num_epochs * train_data_len)
    val_eval_score = 0
    train_pages = list(page_sec_paras.keys())
    for p in tqdm(range(len(train_pages)), desc='Training loop'):
        page = train_pages[p]
        if page not in page_paras.keys() or not is_fit_for_training(page_paras[page], page_sec_paras[page]):
            continue
        paras = page_paras[page]
        duo_sbert_emb.train()
        for sec in page_sec_paras[page].keys():
            if '/' in sec:
                query = ' '.join(sec.split('/')[1:]).replace('enwiki:', '').replace('%20', ' ')
            else:
                query = sec.replace('enwiki:', '').replace('%20', ' ')
            rel_paras = page_sec_paras[page][sec]
            nonrel_paras = [p for p in paras if p not in rel_paras]
            random.shuffle(rel_paras)
            random.shuffle(nonrel_paras)
            rel_first_texts, nonrel_first_texts = {'q': [], 'p1': [], 'p2': []}, {'q': [], 'p1': [], 'p2': []}
            for p1 in rel_paras:
                for p2 in nonrel_paras:
                    rel_first_texts['q'].append(query)
                    rel_first_texts['p1'].append(paratext[p1])
                    rel_first_texts['p2'].append(paratext[p2])
                    nonrel_first_texts['q'].append(query)
                    nonrel_first_texts['p2'].append(paratext[p1])
                    nonrel_first_texts['p1'].append(paratext[p2])
            for b in range(len(rel_first_texts['q']) // batch_size):
                rel_first_query_fet_batch = duo_sbert_emb.query_sbert.tokenize(
                    rel_first_texts['q'][b * batch_size: (b+1) * batch_size])
                rel_first_p1_fet_batch = duo_sbert_emb.psg_sbert.tokenize(
                    rel_first_texts['p1'][b * batch_size: (b + 1) * batch_size])
                rel_first_p2_fet_batch = duo_sbert_emb.psg_sbert.tokenize(
                    rel_first_texts['p2'][b * batch_size: (b + 1) * batch_size])
                nonrel_first_query_fet_batch = duo_sbert_emb.query_sbert.tokenize(
                    nonrel_first_texts['q'][b * batch_size: (b + 1) * batch_size])
                nonrel_first_p1_fet_batch = duo_sbert_emb.psg_sbert.tokenize(
                    nonrel_first_texts['p1'][b * batch_size: (b + 1) * batch_size])
                nonrel_first_p2_fet_batch = duo_sbert_emb.psg_sbert.tokenize(
                    nonrel_first_texts['p2'][b * batch_size: (b + 1) * batch_size])
                put_features_in_device(rel_first_query_fet_batch, device)
                put_features_in_device(rel_first_p1_fet_batch, device)
                put_features_in_device(rel_first_p2_fet_batch, device)
                put_features_in_device(nonrel_first_query_fet_batch, device)
                put_features_in_device(nonrel_first_p1_fet_batch, device)
                put_features_in_device(nonrel_first_p2_fet_batch, device)
                rel_first_preds = duo_sbert_emb(rel_first_query_fet_batch, rel_first_p1_fet_batch,
                                                rel_first_p2_fet_batch)
                nonrel_first_preds = duo_sbert_emb(nonrel_first_query_fet_batch, nonrel_first_p1_fet_batch,
                                                   nonrel_first_p2_fet_batch)
                loss = - torch.sum(torch.log(rel_first_preds)) - torch.sum(torch.log(1 - nonrel_first_preds))
                loss.backward()
                #print('\rLoss: %.4f' % loss.item(), end='')
                nn.utils.clip_grad_norm_(duo_sbert_emb.parameters(), max_grad_norm)
                opt.step()
                opt.zero_grad()
                schd.step()
        if (p+1) % val_step == 0:
            print('\nEvaluating on validation set...')
            val_measures = eval_duo_sbert_emb(duo_sbert_emb, val_page_paras, val_page_sec_paras, val_paratext, val_qrels)
            print('\nValidation eval MAP: %.4f, Rprec: %.4f, nDCG: %.4f' % (val_measures[MAP], val_measures[Rprec],
                                                                      val_measures[nDCG]))
            if val_measures[MAP] > val_eval_score:
                torch.save(duo_sbert_emb, model_out)
                val_eval_score = val_measures[MAP]
    print('\nTraining complete. Evaluating on test set...')
    test_measures = eval_duo_sbert_emb(duo_sbert_emb, test_page_paras, test_page_sec_paras, test_paratext, test_qrels)
    print('\nTest eval MAP: %.4f, Rprec: %.4f, nDCG: %.4f' % (test_measures[MAP], test_measures[Rprec],
                                                                    test_measures[nDCG]))


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA is available and using device: '+str(device))
    else:
        device = torch.device('cpu')
        print('CUDA not available, using device: '+str(device))
    train_duo_sbert_with_clustering('D:\\new_cats_data\\train\\base.train.cbor-without-by1-article.qrels',
                    'D:\\new_cats_data\\train\\base.train.cbor-without-by1-toplevel.qrels',
                    'D:\\new_cats_data\\train\\train_paratext.tsv',
                    'D:\\retrieval_experiments\\val.train.pages.cbor-without-long-article.qrels',
                    'D:\\retrieval_experiments\\val.train.pages.cbor-without-long-toplevel.qrels',
                    'D:\\new_cats_data\\benchmarkY1\\benchmarkY1-train-nodup\\by1train_paratext\\by1train_paratext.tsv',
                    'D:\\new_cats_data\\benchmarkY1\\benchmarkY1-test-nodup\\test.pages.cbor-article.qrels',
                    'D:\\new_cats_data\\benchmarkY1\\benchmarkY1-test-nodup\\test.pages.cbor-toplevel.qrels',
                    'D:\\new_cats_data\\benchmarkY1\\benchmarkY1-test-nodup\\by1test_paratext\\by1test_paratext.tsv',
                    device,
                    'C:\\Users\\suman\\PycharmProjects\\QSClustering\\saved_models\\duo_sbert_with_2clustering_best_eval.model')


if __name__ == '__main__':
    main()