import numpy as np
from util.data import get_article_qrels, get_page_sec_para_dict


def filter_rel(art_qrels, runfile, outfile):
    page_paras = get_article_qrels(art_qrels)
    first_line = True
    with open(outfile, 'w') as out:
        with open(runfile, 'r') as run:
            for l in run:
                q = l.split(' ')[0].split('/')[0]
                p = l.split(' ')[2]
                if p in page_paras[q]:
                    if first_line:
                        out.write(l.strip())
                        first_line = False
                    else:
                        out.write('\n' + l.strip())


def viz_runfile_with_clusters(art_qrels, qrels, runfile, clustering_model):
    page_paras = get_article_qrels(art_qrels)
    true_page_sec_paras = get_page_sec_para_dict(qrels)
    run_page_sec_paras = get_page_sec_para_dict(runfile)