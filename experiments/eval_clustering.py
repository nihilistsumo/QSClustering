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
    with open(paratext_tsv, 'r') as f:
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
        pred_labels_top = model.get_clustering(embeddings, kt)
        clustering_results[page]['toplevel_cluster_idx'] = pred_labels_top
        kh = len(set(true_labels_hier))
        pred_labels_hier = model.get_clustering(embeddings, kh)
        clustering_results[page]['hier_cluster_idx'] = pred_labels_hier
        rand_top = adjusted_rand_score(true_labels_top, pred_labels_top)
        rand_hier = adjusted_rand_score(true_labels_hier, pred_labels_hier)
        rand_score_top += rand_top
        rand_score_hier += rand_hier
    rand_score_top /= len(page_paras.keys())
    rand_score_hier /= len(page_paras.keys())
    return rand_score_top, rand_score_hier, clustering_results