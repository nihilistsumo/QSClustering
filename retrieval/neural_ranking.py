import random
import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import AdamW
from sentence_transformers import models, SentenceTransformer
import ir_measures
from ir_measures import MAP, Rprec, nDCG
from tqdm import tqdm
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
                for i in range(len(cand_set)):
                    pi = cand_set[i]
                    texts_for_pi = []
                    for j in range(len(cand_set)):
                        if i == j:
                            continue
                        pj = cand_set[j]
                        texts_for_pi.append((query, paratext[pi], paratext[pj]))
                    rel_score = np.sum(duo_sbert.encode(texts_for_pi).flatten())
                    f.write(sec + ' 0 ' + pi + ' 0 ' + str(rel_score) + ' val_runid\n')
    qrels_dat = ir_measures.read_trec_qrels(qrels)
    run_dat = ir_measures.read_trec_run('temp.val.run')
    return ir_measures.calc_aggregate([MAP, Rprec, nDCG], qrels_dat, run_dat)


def is_fit_for_training(paras, sec_paras):
    return 10 <= len(paras) <= 200 and len(sec_paras.keys()) > 2


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
                rel_first_fet_batch = duo_sbert.tokenize(rel_first_texts[b * batch_size: (b+1) * batch_size])
                nonrel_first_fet_batch = duo_sbert.tokenize(nonrel_first_texts[b * batch_size: (b+1) * batch_size])
                put_features_in_device(rel_first_fet_batch, device)
                put_features_in_device(nonrel_first_fet_batch, device)
                rel_first_preds = duo_sbert(rel_first_fet_batch)['sentence_embedding']
                nonrel_first_preds = duo_sbert(nonrel_first_fet_batch)['sentence_embedding']
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
                duo_sbert.save(model_out)
                val_eval_score = val_measures[MAP]
    print('\nTraining complete. Evaluating on test set...')
    test_measures = eval_duo_sbert(duo_sbert, test_page_paras, test_page_sec_paras, test_paratext, test_qrels)
    print('\nTest eval MAP: %.4f, Rprec: %.4f, nDCG: %.4f' % (test_measures[MAP], test_measures[Rprec],
                                                                    test_measures[nDCG]))


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA is available and using device: '+str(device))
    else:
        device = torch.device('cpu')
        print('CUDA not available, using device: '+str(device))
    train_duo_sbert('D:\\new_cats_data\\train\\base.train.cbor-without-by1-article.qrels',
                    'D:\\new_cats_data\\train\\base.train.cbor-without-by1-toplevel.qrels',
                    'D:\\new_cats_data\\train\\train_paratext.tsv',
                    'D:\\retrieval_experiments\\val.train.pages.cbor-without-long-article.qrels',
                    'D:\\retrieval_experiments\\val.train.pages.cbor-without-long-toplevel.qrels',
                    'D:\\new_cats_data\\benchmarkY1\\benchmarkY1-train-nodup\\by1train_paratext\\by1train_paratext.tsv',
                    'D:\\new_cats_data\\benchmarkY1\\benchmarkY1-test-nodup\\test.pages.cbor-article.qrels',
                    'D:\\new_cats_data\\benchmarkY1\\benchmarkY1-test-nodup\\test.pages.cbor-toplevel.qrels',
                    'D:\\new_cats_data\\benchmarkY1\\benchmarkY1-test-nodup\\by1test_paratext\\by1test_paratext.tsv',
                    device,
                    'C:\\Users\\suman\\PycharmProjects\\QSClustering\\saved_models\\duo_sbert_best_eval.model')


if __name__ == '__main__':
    main()