import numpy as np
import torch
import time
import random
random.seed(round(time.time()))
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.metrics import adjusted_rand_score

import core.clustering
from util.data import get_rev_qrels, get_article_qrels


def add_linebreak_str(s, w=50):
    res = []
    for i in range((len(s) // w) + 1):
        res.append(s[i * w: i * w + w])
    return '<br>'.join(res)


def plot_3d_vecs(vecs, title, names=None, labels=None):
    if names is not None:
        assert vecs.shape[0] == len(names)
    if labels is not None:
        assert vecs.shape[0] == len(labels)
    pca = PCA(n_components=3)
    vecs_tr = pca.fit_transform(vecs)
    fig = px.scatter_3d(x=vecs_tr[:, 0], y=vecs_tr[:, 1], z=vecs_tr[:, 2], hover_name=names, color=labels, title=title)
    fig.show()


def compare_treccar_models_cv(qsc_model_path, triplet_model_path, dataset_path, test_fold, query=None,
                              emb_model_name='sentence-transformers/all-MiniLM-L6-v2', emb_dim=256, max_num_tokens=128,
                              triplet_margin=5):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA is available and using device: '+str(device))
    else:
        device = torch.device('cpu')
        print('CUDA not available, using device: '+str(device))
    dataset = np.load(dataset_path, allow_pickle=True)[()]['data']
    samples = [s for s in dataset[test_fold]]
    if query is None:
        sample = random.sample(samples, 1)[0]
    else:
        sample = None
        for s in samples:
            if s.q == query:
                sample = s
                break
    qsc_model = core.clustering.QuerySpecificClusteringModel(emb_model_name, emb_dim, device, max_num_tokens)
    qsc_model.load_state_dict(torch.load(qsc_model_path))
    triplet_model = core.clustering.SBERTTripletLossModel(emb_model_name, device, max_num_tokens, triplet_margin)
    triplet_model.load_state_dict(torch.load(triplet_model_path))
    names = [t[:50] for t in sample.para_texts]
    labels = sample.para_labels
    k = len(set(labels))
    qsc_vecs = qsc_model.qp_model.encode(sample.para_texts, convert_to_tensor=True)
    qsc_pred_labels = qsc_model.get_clustering(qsc_vecs, k)
    qsc_rand = adjusted_rand_score(labels, qsc_pred_labels)
    triplet_vecs = triplet_model.emb_model.encode(sample.para_texts, convert_to_tensor=True)
    triplet_pred_labels = triplet_model.get_clustering(triplet_vecs, k)
    triplet_rand = adjusted_rand_score(labels, triplet_pred_labels)
    plot_3d_vecs(qsc_vecs.detach().cpu().numpy(), 'QSC Adj RAND: %.4f' % qsc_rand, names, labels)
    plot_3d_vecs(triplet_vecs.detach().cpu().numpy(), 'Triplet Adj RAND: %.4f' % triplet_rand, names, labels)


def compare_treccar_models_unseen_queries(qsc_model_path, triplet_model_path, art_qrels, top_qrels, paratext_tsv,
                                          query=None, emb_model_name='sentence-transformers/all-MiniLM-L6-v2',
                                          emb_dim=256, max_num_tokens=128, triplet_margin=5):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA is available and using device: '+str(device))
    else:
        device = torch.device('cpu')
        print('CUDA not available, using device: '+str(device))
    page_paras = get_article_qrels(art_qrels)
    rev_para_labels = get_rev_qrels(top_qrels)
    queries = list(page_paras.keys())
    paratext = {}
    with open(paratext_tsv, 'r', encoding='utf8') as f:
        for l in f:
            paratext[l.split('\t')[0]] = l.split('\t')[1].strip()
    if query is None:
        query = queries[random.randint(0, len(queries))]
    elif type(query) == int:
        query = queries[query]
    paras = page_paras[query]
    labels = [rev_para_labels[p] for p in paras]
    texts = [paratext[p] for p in paras]
    qsc_model = core.clustering.QuerySpecificClusteringModel(emb_model_name, emb_dim, device, max_num_tokens)
    qsc_model.load_state_dict(torch.load(qsc_model_path))
    triplet_model = core.clustering.SBERTTripletLossModel(emb_model_name, device, max_num_tokens, triplet_margin)
    triplet_model.load_state_dict(torch.load(triplet_model_path))
    names = [add_linebreak_str(t) for t in texts]
    k = len(set(labels))
    qsc_vecs = qsc_model.qp_model.encode(texts, convert_to_tensor=True)
    qsc_pred_labels = qsc_model.get_clustering(qsc_vecs, k)
    qsc_rand = adjusted_rand_score(labels, qsc_pred_labels)
    triplet_vecs = triplet_model.emb_model.encode(texts, convert_to_tensor=True)
    triplet_pred_labels = triplet_model.get_clustering(triplet_vecs, k)
    triplet_rand = adjusted_rand_score(labels, triplet_pred_labels)
    plot_3d_vecs(qsc_vecs.detach().cpu().numpy(), 'QSC Adj RAND: %.4f' % qsc_rand, names, labels)
    plot_3d_vecs(triplet_vecs.detach().cpu().numpy(), 'Triplet Adj RAND: %.4f' % triplet_rand, names, labels)


'''
compare_treccar_models_cv('D:\\new_cats_data\\QSC_data\\benchmarkY1-train-nodup\\test_results\\qsc_adj_trec_2cv_minilm_ep75_fold1.model',
                          'D:\\new_cats_data\\QSC_data\\benchmarkY1-train-nodup\\test_results\\triplet_trec_2cv_minilm_ep75_fold1.model',
                          'D:\\new_cats_data\\QSC_data\\benchmarkY1-train-nodup\\treccar_clustering_data_by1train_2cv.npy',
                          1,
                          'enwiki:Allergy')
'''
compare_treccar_models_unseen_queries('D:\\new_cats_data\\QSC_data\\benchmarkY1-train-nodup\\test_results\\qsc_adj_trec_2cv_minilm_ep75_fold1.model',
                                      'D:\\new_cats_data\\QSC_data\\benchmarkY1-train-nodup\\test_results\\triplet_trec_2cv_minilm_ep75_fold1.model',
                                      'D:\\new_cats_data\\QSC_data\\benchmarkY1-test-nodup\\test.pages.cbor-article.qrels',
                                      'D:\\new_cats_data\\QSC_data\\benchmarkY1-test-nodup\\test.pages.cbor-toplevel.qrels',
                                      'D:\\new_cats_data\\QSC_data\\benchmarkY1-test-nodup\\by1test_paratext\\by1test_paratext.tsv',
                                      50)