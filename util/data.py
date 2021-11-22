import json
import math
from torch.utils.data import Dataset
from rank_bm25 import BM25Okapi


def get_article_qrels(art_qrels):
    qparas = {}
    with open(art_qrels, 'r', encoding='utf-8') as f:
        for l in f:
            q = l.split()[0].strip()
            p = l.split()[2].strip()
            if q not in qparas.keys():
                qparas[q] = [p]
            else:
                qparas[q].append(p)
    return qparas


def get_rev_qrels(qrels):
    para_labels = {}
    with open(qrels, 'r', encoding='utf-8') as f:
        for l in f:
            q = l.split()[0].strip()
            p = l.split()[2].strip()
            para_labels[p] = q
    return para_labels


def remove_by1_from_train(train_art_qrels, train_qrels, by1train_art_qrels, by1test_art_qrels, train_art_qrels_out,
                          train_qrels_out):
    psgs_to_remove = set()
    articles_to_remove = set()
    with open(by1train_art_qrels, 'r') as f:
        for l in f:
            psgs_to_remove.add(l.split(' ')[2])
            articles_to_remove.add(l.split(' ')[0])
    with open(by1test_art_qrels, 'r') as f:
        for l in f:
            psgs_to_remove.add(l.split(' ')[2])
            articles_to_remove.add(l.split(' ')[0])
    with open(train_art_qrels_out, 'w') as f:
        with open(train_art_qrels, 'r') as r:
            for l in r:
                if l.split(' ')[0] in articles_to_remove or l.split(' ')[2] in psgs_to_remove:
                    print('Removing ' + l)
                else:
                    f.write(l.strip() + '\n')
    with open(train_qrels_out, 'w') as f:
        with open(train_qrels, 'r') as r:
            for l in r:
                if l.split(' ')[0] in articles_to_remove or l.split(' ')[0].split('/')[0] in articles_to_remove or l.split(' ')[2] in psgs_to_remove:
                    print('Removing ' + l)
                else:
                    f.write(l.strip() + '\n')


class TRECCAR_sample(object):
    def __init__(self, q, paras, para_labels, para_texts):
        self.q = q
        self.paras = paras
        self.para_labels = para_labels
        self.para_texts = para_texts

    def __str__(self):
        return 'Sample ' + self.q + ' with %d passages and %d unique clusters' % (len(self.paras),
                                                                                  len(set(self.para_labels)))


class TRECCAR_Datset(Dataset):
    def __init__(self, article_qrels, qrels, paratext_tsv, max_num_paras, selected_enwiki_titles=None):
        if selected_enwiki_titles is not None:
            q_paras = get_article_qrels(article_qrels)
            self.q_paras = {}
            for q in q_paras.keys():
                if q in selected_enwiki_titles:
                    self.q_paras[q] = q_paras[q]
            del q_paras
        else:
            self.q_paras = get_article_qrels(article_qrels)
        paras = []
        for q in self.q_paras.keys():
            paras += self.q_paras[q]
        self.paras = set(paras)
        self.queries = list(self.q_paras.keys())
        self.rev_para_labels = get_rev_qrels(qrels)
        self.paratext = {}
        with open(paratext_tsv, 'r', encoding='utf-8') as f:
            for l in f:
                p = l.split('\t')[0]
                if p in self.paras:
                    self.paratext[p] = l.split('\t')[1].strip()
        self.max_num_paras = max_num_paras

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        q = self.queries[idx]
        paras = self.q_paras[q]
        para_labels = [self.rev_para_labels[p] for p in paras]
        para_texts = [self.paratext[p] for p in paras]
        num_paras = len(paras)
        num_batches = math.ceil(num_paras / self.max_num_paras)
        batched_samples = []
        for b in range(num_batches):
            paras_batch = paras[b * self.max_num_paras: (b + 1) * self.max_num_paras]
            para_labels_batch = para_labels[b * self.max_num_paras: (b + 1) * self.max_num_paras]
            n = len(paras_batch)
            k = len(set(para_labels_batch))
            if k > 1 and k < n:
                batched_samples.append(TRECCAR_sample(q, paras_batch, para_labels_batch,
                                                      para_texts[b * self.max_num_paras: (b + 1) * self.max_num_paras]))
        return batched_samples


class Vital_Wiki_Dataset(Dataset):
    def __init__(self, vital_cats_data, q_paras, rev_para_qrels, paratext_tsv, max_num_paras):
        self.q_paras = q_paras
        self.vital_cats = vital_cats_data
        self.articles = []
        self.paras = []
        self.rev_art_cat = {}
        for cat in self.vital_cats.keys():
            for a in self.vital_cats[cat]:
                enwiki_art = 'enwiki:' + a
                if enwiki_art in self.q_paras.keys():
                    self.articles.append(enwiki_art)
                    self.rev_art_cat[enwiki_art] = cat
                    self.paras += self.q_paras[enwiki_art]
                else:
                    print(enwiki_art + ' missing in qrels')
        self.paras = set(self.paras)
        self.rev_para_labels = rev_para_qrels
        self.paratext = {}
        with open(paratext_tsv, 'r', encoding='utf-8') as f:
            for l in f:
                p = l.split('\t')[0]
                if p in self.paras:
                    self.paratext[p] = l.split('\t')[1].strip()
        self.max_num_paras = max_num_paras

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        a = self.articles[idx]
        paras = self.q_paras[a]
        para_labels = [self.rev_para_labels[p] for p in paras]
        para_texts = [self.paratext[p] for p in paras]
        num_paras = len(paras)
        num_batches = math.ceil(num_paras / self.max_num_paras)
        batched_samples = []
        for b in range(num_batches):
            paras_batch = paras[b * self.max_num_paras: (b + 1) * self.max_num_paras]
            para_labels_batch = para_labels[b * self.max_num_paras: (b + 1) * self.max_num_paras]
            n = len(paras_batch)
            k = len(set(para_labels_batch))
            if k > 1 and k < n:
                batched_samples.append(TRECCAR_sample(self.rev_art_cat[a], paras_batch, para_labels_batch,
                                                      para_texts[b * self.max_num_paras: (b + 1) * self.max_num_paras]))
        return batched_samples


def separate_multi_batch(sample, max_num_paras):
    num_paras = len(sample.paras)
    num_batches = math.ceil(num_paras / max_num_paras)
    batched_samples = []
    for b in range(num_batches):
        paras = sample.paras[b * max_num_paras: (b+1) * max_num_paras]
        para_labels = sample.para_labels[b * max_num_paras: (b+1) * max_num_paras]
        n = len(paras)
        k = len(set(para_labels))
        if k > 1 and k < n:
            batched_samples.append(TRECCAR_sample(sample.q, paras, para_labels,
                                                  sample.para_texts[b * max_num_paras: (b+1) * max_num_paras]))
    return batched_samples


def get_similar_treccar_articles(query_enwiki_titles, article_qrels, n):
    art_qrels = get_article_qrels(article_qrels)
    enwiki_map = {}
    for q in art_qrels.keys():
        enwiki_map[q.split('enwiki:')[1].replace('%20', ' ')] = q
    corpus = list(enwiki_map.keys())
    tokenized_corpus = [q.split(' ') for q in corpus]
    query_titles = [q.split('enwiki:')[1].replace('%20', ' ') for q in query_enwiki_titles]
    bm25 = BM25Okapi(tokenized_corpus)
    result = []
    print('Finding similar articles')
    for q in query_titles:
        result += [enwiki_map[a] for a in bm25.get_top_n(q.split(' '), corpus, n)
                   if enwiki_map[a] not in query_enwiki_titles]
    return result