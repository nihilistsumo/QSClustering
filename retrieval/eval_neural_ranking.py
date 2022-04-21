import os
import argparse
import torch
from ir_measures import AP, nDCG, Rprec
from neural_ranking import eval_mono_bert_ranking_full, prepare_data, Mono_SBERT_Clustering_Reg_Model


def do_eval(model_dir, art_qrels, qrels, paratext_tsv):
    page_paras, page_sec_paras, paratext = prepare_data(art_qrels, qrels, paratext_tsv)
    for mname in os.listdir(model_dir):
        model = torch.load(model_dir + '\\' + mname)
        rank_eval = eval_mono_bert_ranking_full(model, page_paras, page_sec_paras, paratext, qrels, True)
        results_per_query = {}
        for metric in rank_eval:
            q = metric.query_id
            if q in results_per_query.keys():
                results_per_query[q][metric.measure] = metric.value
            else:
                results_per_query[q] = {metric.measure: metric.value}
        print('\n' + mname)
        for q in results_per_query.keys():
            print('%s,%.4f,%.4f,%.4f' % (q, results_per_query[q][AP], results_per_query[q][nDCG], results_per_query[q][Rprec]))


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA is available and using device: '+str(device))
    else:
        device = torch.device('cpu')
        print('CUDA not available, using device: '+str(device))
    parser = argparse.ArgumentParser(description='Neural ranking evaluation')
    parser.add_argument('-va', '--val_art_qrels',
                        default='D:\\new_cats_data\\benchmarkY1\\benchmarkY1-train-nodup\\train.pages.cbor-article.qrels')
    parser.add_argument('-vq', '--val_qrels',
                        default='D:\\new_cats_data\\benchmarkY1\\benchmarkY1-train-nodup\\train.pages.cbor-toplevel.qrels')
    parser.add_argument('-vp', '--val_ptext',
                        default='D:\\new_cats_data\\benchmarkY1\\benchmarkY1-train-nodup\\by1train_paratext\\by1train_paratext.tsv')
    parser.add_argument('-ta', '--test_art_qrels',
                        default='D:\\new_cats_data\\benchmarkY1\\benchmarkY1-test-nodup\\test.pages.cbor-article.qrels')
    parser.add_argument('-tq', '--test_qrels',
                        default='D:\\new_cats_data\\benchmarkY1\\benchmarkY1-test-nodup\\test.pages.cbor-toplevel.qrels')
    parser.add_argument('-tp', '--test_ptext',
                        default='D:\\new_cats_data\\benchmarkY1\\benchmarkY1-test-nodup\\by1test_paratext\\by1test_paratext.tsv')
    parser.add_argument('-md', '--model_dir', default='D:\\retrieval_experiments\\monobert_clustering_reg\\saved_models')
    parser.add_argument('-ne', '--number_exp', type=int, default=1)
    parser.add_argument('-mn', '--model_name', help='SBERT embedding model name', default='sentence-transformers/all-MiniLM-L6-v2')

    args = parser.parse_args()
    print('Validation results')
    print('==================')
    do_eval(args.model_dir, args.val_art_qrels, args.val_qrels, args.val_ptext)
    print('\nTest results')
    print('============')
    do_eval(args.model_dir, args.test_art_qrels, args.test_qrels, args.test_ptext)


if __name__ == '__main__':
    main()