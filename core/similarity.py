import numpy as np
import torch
import random
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering
from core.clustering import SBERTTripletLossModel, QuerySpecificClusteringModel, DKM


def vital_quick_eval(model_path, data_path, model_fold, model_name='sentence-transformers/all-MiniLM-L6-v2',
                     emb_dim=256, max_len=128):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA is available and using device: '+str(device))
    else:
        device = torch.device('cpu')
        print('CUDA not available, using device: '+str(device))
    data = np.load(data_path, allow_pickle=True)[()]['data']
    test_data = data[model_fold].test_samples
    model = QuerySpecificClusteringModel(model_name, emb_dim, device, max_len)
    model.load_state_dict(torch.load(model_path))
    dkm = DKM()
    model.eval()
    rand_dict = {}
    for s in test_data:
        print(s.q)
        true_labels = s.para_labels
        n = len(s.para_texts)
        k = len(set(true_labels))
        texts = s.para_texts
        query_content = s.category
        input_texts = [(query_content, t) for t in texts]
        embeddings = model.qp_model.encode(input_texts, convert_to_tensor=True)
        #c, a = dkm(embeddings, embeddings[random.sample(range(n), k)])

        #dist_mat = torch.cdist(a, a).detach().cpu().numpy()
        #sim_mat = 1 / (1 + dist_mat)
        #cl = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='average')
        #pred_labels = cl.fit_predict(sim_mat)
        pred_labels = model.get_clustering(embeddings, k, True)

        rand = adjusted_rand_score(true_labels, pred_labels)
        rand_dict[s.q] = rand
    print('Mean Adj. RAND: %.4f +- %.4f' % (np.mean(list(rand_dict.values())),
                                            np.std(list(rand_dict.values()), ddof=1) / np.sqrt(len(rand_dict.keys()))))


vital_quick_eval('D:\\new_cats_data\\QSC_data\\vital_wiki\\test_results\\qsc_adj_vital_wiki_fold1.model',
                 'D:\\new_cats_data\\QSC_data\\vital_wiki\\vital_wiki_clustering_data_2cv.npy', 0)