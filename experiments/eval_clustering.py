import json
import numpy as np
from sklearn.metrics import adjusted_rand_score
from urllib.parse import unquote


def do_eval(art_qrels, top_qrels, hier_qrels, paratext_tsv, model):
    model.eval()
    rand_score_top, rand_score_hier = 0, 0
    page_paras = {}
    with open(art_qrels, 'r') as f:
        for l in f:
            page = l.split(' ')[0]
            para = l.split(' ')[2]
            if page in page_paras.keys():
                page_paras[page].append(para)
            else:
                page_paras[page] = [para]
    rev_para_sec_labels_top = {}
    rev_para_sec_labels_hier = {}
    with open(top_qrels, 'r') as f:
        for l in f:
            rev_para_sec_labels_top[l.split(' ')[2]] = l.split(' ')[0]
    with open(hier_qrels, 'r') as f:
        for l in f:
            rev_para_sec_labels_hier[l.split(' ')[2]] = l.split(' ')[0]
    paratexts = {}
    with open(paratext_tsv, 'r', encoding='utf8') as f:
        for l in f:
            paratexts[l.split('\t')[0]] = l.split('\t')[1].strip()
    clustering_results = {}
    for page in page_paras.keys():
        clustering_results[page] = {'query_id': page, 'query_text': unquote(page.split('enwiki:')[1])}
        true_labels_top = [rev_para_sec_labels_top[p] for p in page_paras[page]]
        true_labels_hier = [rev_para_sec_labels_hier[p] for p in page_paras[page]]
        paras = page_paras[page]
        clustering_results[page]['elements'] = paras
        texts = [paratexts[p] for p in page_paras[page]]
        query_content = page.split('enwiki:')[1].replace('%20', ' ')
        input_texts = [(query_content, t) for t in texts]
        embeddings = model.qp_model.encode(input_texts, convert_to_tensor=True)
        kt = len(set(true_labels_top))
        pred_labels_top = model.get_clustering(embeddings, kt).tolist()
        clustering_results[page]['toplevel_cluster_idx'] = pred_labels_top
        kh = len(set(true_labels_hier))
        pred_labels_hier = model.get_clustering(embeddings, kh).tolist()
        clustering_results[page]['hier_cluster_idx'] = pred_labels_hier
        rand_top = adjusted_rand_score(true_labels_top, pred_labels_top)
        rand_hier = adjusted_rand_score(true_labels_hier, pred_labels_hier)
        rand_score_top += rand_top
        rand_score_hier += rand_hier
        print(page + ' top ARI: %.4f, hier ARI: %.4f' % (rand_top, rand_hier))
    rand_score_top /= len(page_paras.keys())
    rand_score_hier /= len(page_paras.keys())
    return rand_score_top, rand_score_hier, clustering_results


def get_clusters_from_cand_run(cand_run_qrels_format, art_qrels, top_qrels, hier_qrels, paratext_tsv, model):
    model.eval()
    page_paras = {}
    with open(cand_run_qrels_format, 'r') as f:
        for l in f:
            page = l.split(' ')[0]
            para = l.split(' ')[2]
            if page in page_paras.keys():
                page_paras[page].append(para)
            else:
                page_paras[page] = [para]
    rev_para_sec_labels_top = {}
    rev_para_sec_labels_hier = {}
    pages_nodup = set()
    with open(art_qrels, 'r') as f:
        for l in f:
            pages_nodup.add(l.split(' ')[0])
    with open(top_qrels, 'r') as f:
        for l in f:
            rev_para_sec_labels_top[l.split(' ')[2]] = l.split(' ')[0]
    with open(hier_qrels, 'r') as f:
        for l in f:
            rev_para_sec_labels_hier[l.split(' ')[2]] = l.split(' ')[0]
    paratexts = {}
    with open(paratext_tsv, 'r', encoding='utf8') as f:
        for l in f:
            paratexts[l.split('\t')[0]] = l.split('\t')[1].strip()
    clustering_results = {}
    for page in page_paras.keys():
        if page not in pages_nodup:
            print(page + ' not present in nodup art qrels')
            continue
        clustering_results[page] = {'query_id': page, 'query_text': unquote(page.split('enwiki:')[1])}
        true_labels_top = [rev_para_sec_labels_top[p] if p in rev_para_sec_labels_top.keys() else 'nonrel' for p in page_paras[page]]
        true_labels_hier = [rev_para_sec_labels_hier[p] if p in rev_para_sec_labels_hier.keys() else 'nonrel' for p in page_paras[page]]
        paras = page_paras[page]
        clustering_results[page]['elements'] = paras
        texts = [paratexts[p] for p in page_paras[page]]
        query_content = page.split('enwiki:')[1].replace('%20', ' ')
        input_texts = [(query_content, t) for t in texts]
        embeddings = model.qp_model.encode(input_texts, convert_to_tensor=True)
        kt = len(set(true_labels_top))
        pred_labels_top = model.get_clustering(embeddings, kt).tolist()
        clustering_results[page]['toplevel_cluster_idx'] = pred_labels_top
        kh = len(set(true_labels_hier))
        pred_labels_hier = model.get_clustering(embeddings, kh).tolist()
        clustering_results[page]['hier_cluster_idx'] = pred_labels_hier
        print(page)
    return clustering_results


def eval_clustering_json(clustering_results_file, top_qrels, hier_qrels):
    with open(clustering_results_file, 'r') as f:
        cl = json.load(f)
    top_ari, hier_ari = [], []
    rev_para_sec_labels_top = {}
    rev_para_sec_labels_hier = {}
    with open(top_qrels, 'r') as f:
        for l in f:
            rev_para_sec_labels_top[l.split(' ')[2]] = l.split(' ')[0]
    with open(hier_qrels, 'r') as f:
        for l in f:
            rev_para_sec_labels_hier[l.split(' ')[2]] = l.split(' ')[0]
    for page in cl.keys():
        paras = cl[page]['elements']
        pred_top = cl[page]['toplevel_cluster_idx']
        pred_hier = cl[page]['hier_cluster_idx']
        true_top = [rev_para_sec_labels_top[p] for p in paras]
        true_hier = [rev_para_sec_labels_hier[p] for p in paras]
        top_ari.append(adjusted_rand_score(true_top, pred_top))
        hier_ari.append(adjusted_rand_score(true_hier, pred_hier))
    print(clustering_results_file.split('\\')[len(clustering_results_file.split('\\'))-1] + ' Toplevel ARI: %.4f +- %.4f, Hier level ARI: %.4f +- %.4f' % (np.sum(top_ari)/len(cl.keys()),
                                                                        np.std(top_ari)/np.sqrt(len(cl.keys())),
                                                                        np.sum(hier_ari)/len(cl.keys()),
                                                                        np.std(hier_ari)/np.sqrt(len(cl.keys()))))