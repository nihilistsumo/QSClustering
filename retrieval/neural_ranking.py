import random
import torch
import torch.nn as nn
import transformers
from transformers import AdamW
from sentence_transformers import models, SentenceTransformer
from util.data import get_article_qrels, get_page_sec_para_dict
from experiments.treccar_clustering import put_features_in_device
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


def train_duo_sbert(art_qrels,
                    qrels,
                    paratext_tsv,
                    test_art_qrels,
                    test_qrels,
                    test_paratext_tsv,
                    device,
                    trans_model_name='sentence-transformers/all-MiniLM-L6-v2',
                    max_len=512,
                    max_grad_norm=1.0,
                    weight_decay=0.01,
                    warmup=10000,
                    lrate=2e-5,
                    num_epochs=3,
                    batch_size=8):
    page_paras, page_sec_paras, paratext = prepare_data(art_qrels, qrels, paratext_tsv)
    test_page_paras, test_page_sec_paras, test_paratext = prepare_data(test_art_qrels, test_qrels, test_paratext_tsv)
    trans_model = models.Transformer(trans_model_name, max_seq_length=max_len)
    pool_model = models.Pooling(trans_model.get_word_embedding_dimension())
    dense_model = models.Dense(in_features=pool_model.get_sentence_embedding_dimension(), out_features=1,
                               activation_function=nn.Sigmoid())
    duo_sbert = SentenceTransformer(modules=[trans_model, pool_model, dense_model]).to(device)
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
    for page in page_sec_paras.keys():
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
                rel_first_fet_batch = duo_sbert.tokenize(rel_first_texts[b * batch_size: (b+1) * batch_size])
                nonrel_first_fet_batch = duo_sbert.tokenize(nonrel_first_texts[b * batch_size: (b+1) * batch_size])
                put_features_in_device(rel_first_fet_batch, device)
                put_features_in_device(nonrel_first_fet_batch, device)
                rel_first_preds = duo_sbert(rel_first_fet_batch)['sentence_embedding']
                nonrel_first_preds = duo_sbert(nonrel_first_fet_batch)['sentence_embedding']
                loss = - torch.sum(torch.log(rel_first_preds)) - torch.sum(torch.log(1 - nonrel_first_preds))
                loss.backward()
                print('Loss: %.4f, Page: %s, Query: %s, Batch: %4d' % (loss.item(), page, query, b+1))
                nn.utils.clip_grad_norm_(duo_sbert.parameters(), max_grad_norm)
                opt.step()
                opt.zero_grad()
                schd.step()


if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available and using device: '+str(device))
else:
    device = torch.device('cpu')
    print('CUDA not available, using device: '+str(device))
train_duo_sbert('D:\\new_cats_data\\benchmarkY1\\benchmarkY1-train-nodup\\train.pages.cbor-article.qrels',
                'D:\\new_cats_data\\benchmarkY1\\benchmarkY1-train-nodup\\train.pages.cbor-toplevel.qrels',
                'D:\\new_cats_data\\benchmarkY1\\benchmarkY1-train-nodup\\by1train_paratext\\by1train_paratext.tsv',
                device)