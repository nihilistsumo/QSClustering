import numpy as np
import pandas as pd
import torch
import time
import random
random.seed(round(time.time()))
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

import core.clustering
from util.data import get_rev_qrels, get_article_qrels
from experiments.arxiv_abstract_clustering import default_subject_desc


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


def query_effect_vital_wiki(vital_wiki_2cv_data, model_path, model_train_fold, num_run=1,
                            emb_model_name='sentence-transformers/all-MiniLM-L6-v2', emb_dim=256, max_num_tokens=128):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA is available and using device: '+str(device))
    else:
        device = torch.device('cpu')
        print('CUDA not available, using device: '+str(device))
    model = core.clustering.QuerySpecificClusteringModel(emb_model_name, emb_dim, device, max_num_tokens)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    dataset = np.load(vital_wiki_2cv_data, allow_pickle=True)[()]['data']
    test_samples = dataset[model_train_fold].test_samples
    cat_art_map = {}
    for s in test_samples:
        if s.category not in cat_art_map.keys():
            cat_art_map[s.category] = [s]
        else:
            cat_art_map[s.category].append(s)
    categories = list(cat_art_map.keys())
    adj_scores = np.zeros((len(categories), len(categories)))
    for n in tqdm(range(num_run)):
        sampled_articles = []
        for c in categories:
            sampled_articles.append(random.sample(cat_art_map[c], 1)[0])
        for i in range(len(categories)):
            for j in range(len(sampled_articles)):
                cat = categories[i]
                sample = sampled_articles[j]
                k = len(set(sample.para_labels))
                input_texts = [(cat, t) for t in sample.para_texts]
                embeddings = model.qp_model.encode(input_texts, convert_to_tensor=True)
                rand = adjusted_rand_score(sample.para_labels, model.get_clustering(embeddings, k))
                adj_scores[i][j] += rand
    adj_scores = adj_scores / num_run
    adj_scores = (adj_scores - np.min(adj_scores, axis=0)) / (np.max(adj_scores, axis=0) - np.min(adj_scores, axis=0))
    df = pd.DataFrame(adj_scores, index=categories, columns=[c + ' article' for c in categories])
    fig = px.imshow(df)
    fig.update_xaxes(side="top")
    fig.show()


def query_effect_arxiv(arxiv_5cv_data, model_path, model_train_fold, subject_desc,
                            emb_model_name='sentence-transformers/all-MiniLM-L6-v2', emb_dim=256, max_num_tokens=128):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA is available and using device: '+str(device))
    else:
        device = torch.device('cpu')
        print('CUDA not available, using device: '+str(device))
    model = core.clustering.QuerySpecificClusteringModel(emb_model_name, emb_dim, device, max_num_tokens)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    dataset = np.load(arxiv_5cv_data, allow_pickle=True)[()]['data']
    test_data = dataset[model_train_fold].test_arxiv_data
    subjects = list(test_data.keys())
    adj_scores = np.zeros((len(subjects), len(subjects)))
    for i in tqdm(range(len(subjects))):
        for j in range(len(subjects)):
            query_content = subject_desc[subjects[i]]
            texts = [d['abstract'] for d in test_data[subjects[j]]]
            true_labels = [d['label'] for d in test_data[subjects[j]]]
            k = len(set(true_labels))
            input_texts = [(query_content, t) for t in texts]
            embeddings = model.qp_model.encode(input_texts, convert_to_tensor=True)
            rand = adjusted_rand_score(true_labels, model.get_clustering(embeddings, k))
            adj_scores[i][j] = rand
    df = pd.DataFrame(adj_scores, index=subjects, columns=[c + ' abstracts' for c in subjects])
    fig = px.imshow(df)
    fig.update_xaxes(side="top")
    fig.show()


'''
compare_treccar_models_cv('D:\\new_cats_data\\QSC_data\\benchmarkY1-train-nodup\\test_results\\qsc_adj_trec_2cv_minilm_ep75_fold1.model',
                          'D:\\new_cats_data\\QSC_data\\benchmarkY1-train-nodup\\test_results\\triplet_trec_2cv_minilm_ep75_fold1.model',
                          'D:\\new_cats_data\\QSC_data\\benchmarkY1-train-nodup\\treccar_clustering_data_by1train_2cv.npy',
                          1,
                          'enwiki:Allergy')

compare_treccar_models_unseen_queries('D:\\new_cats_data\\QSC_data\\benchmarkY1-train-nodup\\test_results\\qsc_adj_trec_2cv_minilm_ep75_fold1.model',
                                      'D:\\new_cats_data\\QSC_data\\benchmarkY1-train-nodup\\test_results\\triplet_trec_2cv_minilm_ep75_fold1.model',
                                      'D:\\new_cats_data\\QSC_data\\benchmarkY1-test-nodup\\test.pages.cbor-article.qrels',
                                      'D:\\new_cats_data\\QSC_data\\benchmarkY1-test-nodup\\test.pages.cbor-toplevel.qrels',
                                      'D:\\new_cats_data\\QSC_data\\benchmarkY1-test-nodup\\by1test_paratext\\by1test_paratext.tsv',
                                      50)

query_effect_vital_wiki('D:\\new_cats_data\\QSC_data\\vital_wiki\\vital_wiki_clustering_data_2cv.npy',
                        'D:\\new_cats_data\\QSC_data\\vital_wiki\\test_results\\qsc_adj_vital_wiki_fold1.model', 0, 40)
'''
query_effect_arxiv('D:\\arxiv_dataset\\arxiv_clustering_data_5cv.npy',
                   'D:\\arxiv_dataset\\test_results\\arxiv_qsc_adj_ep5_fold1.model', 0, default_subject_desc)