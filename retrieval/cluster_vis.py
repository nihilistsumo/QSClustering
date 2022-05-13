import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from retrieval.neural_ranking import prepare_data, Mono_SBERT_Clustering_Reg_Model


def cov(X):
    return np.dot(X.T, X) / X.shape[0]

def pca(data, pc_count = None):
    data -= np.mean(data, 0)
    data /= np.std(data, 0)
    C = cov(data)
    E, V = np.linalg.eigh(C)
    key = np.argsort(E)[::-1][:pc_count]
    E, V = E[key], V[:, key]
    U = np.dot(data, V)  # used to be dot(V.T, data.T).T
    return U, E, V

def viz_cluster(page, emb_dict_path, s=5):
    emb_dict = np.load(emb_dict_path, allow_pickle=True)
    vecs = emb_dict[()][page]['vec']
    labels = emb_dict[()][page]['label']
    print(vecs.shape)
    pca_embeddings = pca(vecs, 2)[0]
    ax = sns.scatterplot(x=pca_embeddings[:, 0], y=pca_embeddings[:, 1], hue=labels, palette='deep', s=s)
    '''
    ax.legend([], [], frameon=False)
    ax.set(xticklabels=[])
    ax.set(xlabel=None)
    ax.set(yticklabels=[])
    ax.set(ylabel=None)
    ax.tick_params(bottom=False)
    '''
    return ax

def compare_viz_cluster(pages, emb_dict_path1, emb_dict_path2, save_dir_path, s=5, xsize=50, ysize=25):
    emb_dict1 = np.load(emb_dict_path1, allow_pickle=True)
    emb_dict2 = np.load(emb_dict_path2, allow_pickle=True)
    for i in range(len(pages)):
        print(pages[i])
        vecs1 = emb_dict1[()][pages[i]]['vec']
        labels1 = emb_dict1[()][pages[i]]['label']
        pca1 = pca(vecs1, 2)[0]
        vecs2 = emb_dict2[()][pages[i]]['vec']
        labels2 = emb_dict2[()][pages[i]]['label']
        pca2 = pca(vecs2, 2)[0]
        fig, axes = plt.subplots(1, 2, figsize=(xsize, ysize))
        fig.suptitle(pages[i])
        sns.scatterplot(ax=axes[0], x=pca1[:, 0], y=pca1[:, 1], hue=labels1, palette='deep', s=s)
        sns.scatterplot(ax=axes[1], x=pca2[:, 0], y=pca2[:, 1], hue=labels2, palette='deep', s=s)
        plt.savefig(save_dir_path+'/'+pages[i]+'.png')

def compare_page_cluster(vecs1, vecs2, labels, s=300, xsize=50, ysize=25, dim=2):
    pca1 = pca(vecs1, dim)[0]
    pca2 = pca(vecs2, dim)[0]

    if dim == 2:
        fig, axes = plt.subplots(1, 2, figsize=(xsize, ysize))
        sns.scatterplot(ax=axes[0], x=pca1[:, 0], y=pca1[:, 1], hue=labels, palette='deep', s=s)
        sns.scatterplot(ax=axes[1], x=pca2[:, 0], y=pca2[:, 1], hue=labels, palette='deep', s=s)
    elif dim ==3:
        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111, projection='3d')
        ax1.scatter(pca1[:, 0], pca1[:, 1], pca1[:, 2], c=labels, s=s, edgecolors='b')
        fig2 = plt.figure(2)
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.scatter(pca2[:, 0], pca2[:, 1], pca2[:, 2], c=labels, s=s, edgecolors='b')
        axes = [ax1, ax2]
    for ax in axes:
        ax.legend([], [], frameon=False)
        ax.set(xticklabels=[])
        ax.set(xlabel=None)
        ax.set(yticklabels=[])
        ax.set(ylabel=None)
        if dim == 3:
            ax.set(zticklabels=[])
            ax.set(zlabel=None)
        ax.tick_params(bottom=False)
    plt.show()


def run_viz(model1_path, model2_path, art_qrels, qrels, paratext, page, query, dim):
    m1 = torch.load(model1_path)
    m2 = torch.load(model2_path)
    page_paras, page_sec_paras, paratext = prepare_data(art_qrels, qrels, paratext)
    paras_in_page = []
    labels = []
    label = 0
    for s in page_sec_paras[page].keys():
        paras_in_page += page_sec_paras[page][s]
        labels += [label] * len(page_sec_paras[page][s])
        label += 1
    paratexts_in_page = [paratext[p] for p in paras_in_page]
    paravec_m1 = m1.get_qs_embeddings(query, paratexts_in_page).detach().cpu().numpy()
    paravec_m2 = m2.get_qs_embeddings(query, paratexts_in_page).detach().cpu().numpy()
    compare_page_cluster(paravec_m1, paravec_m2, labels, dim=dim)
