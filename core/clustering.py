import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import models, SentenceTransformer
from sklearn.cluster import KMeans
import random
random.seed(42)


# Both gt and a are n x k matrix
def get_nmi_loss(a, para_labels, device):
    n = len(para_labels)
    unique_labels = list(set(para_labels))
    k = len(unique_labels)
    gt = torch.zeros(n, k, device=device)
    for i in range(n):
        gt[i][unique_labels.index(para_labels[i])] = 1.0
    G = torch.sum(gt, dim=0)
    C = torch.sum(a, dim=0)
    U = torch.matmul(gt.T, a)
    n = torch.sum(a)
    GxC = torch.outer(G, C)
    mi = torch.sum((U / n) * torch.log(n * U / GxC))
    nmi = 2 * mi / (-torch.sum(G * torch.log(G / n) / n) - torch.sum(C * torch.log(C / n) / n))
    return -nmi


def get_weighted_adj_rand_loss(a, para_labels, device):
    n = len(para_labels)
    unique_labels = list(set(para_labels))
    gt = torch.zeros(n, n, device=device)
    gt_weights = torch.ones(n, n, device=device)
    para_label_freq = {k: para_labels.count(k) for k in unique_labels}
    for i in range(n):
        for j in range(n):
            if para_labels[i] == para_labels[j]:
                gt[i][j] = 1.0
                gt_weights[i][j] = para_label_freq[para_labels[i]]
    sim_mat = 1 / (1 + torch.cdist(a, a))
    loss = torch.sum(((gt - sim_mat) ** 2) * gt_weights) / gt.shape[0]
    return loss


class DKM(nn.Module):

    def __init__(self, temp=0.5, threshold=0.0001, max_iter=100, eps=1e-6):
        super(DKM, self).__init__()
        self.temp = temp
        self.threshold = threshold
        self.max_iter = max_iter
        self.softmax = nn.Softmax(dim=1)
        self.cos = nn.CosineSimilarity()
        self.eps = eps

    def cosine_sim(self, x, y):
        return self.cos(x.repeat_interleave(y.shape[0], dim=0), y.repeat((x.shape[0], 1))).reshape(x.shape[0], y.shape[0])

    def forward(self, X, C_init):
        self.emb_dim = X.shape[1]
        self.C = C_init
        self.d = -torch.cdist(X, C_init, p=2.0)
        #self.d = self.cosine_sim(X, C_init)
        self.a = self.softmax(self.d/self.temp)
        self.a_sum = torch.sum(self.a, dim=0) + self.eps
        self.C_new = torch.matmul(self.a.T, X)/self.a_sum.repeat((self.emb_dim, 1)).T
        diff = torch.norm(self.C_new - self.C, p=1).item()
        i = 0
        while diff > self.threshold and i < self.max_iter:
            self.C = self.C_new
            self.d = -torch.cdist(X, self.C, p=2.0)
            #self.d = self.cosine_sim(X, C_init)
            self.a = self.softmax(self.d / self.temp)
            self.a_sum = torch.sum(self.a, dim=0) + self.eps
            self.C_new = torch.matmul(self.a.T, X) / self.a_sum.repeat((self.emb_dim, 1)).T
            diff = torch.norm(self.C_new - self.C, p=1).item()
            i += 1
        return self.C, self.a


class DKM_param(nn.Module):

    def __init__(self, temp=0.05, threshold=0.0001, max_iter=100, eps=1e-6, emb_dim=768, l1_size=512):
        super(DKM_param, self).__init__()
        self.temp = temp
        self.threshold = threshold
        self.max_iter = max_iter
        self.emb_dim = emb_dim
        self.softmax = nn.Softmax(dim=1)
        self.l1_size = l1_size
        self.fc1 = nn.Linear(2 * self.emb_dim, self.l1_size)
        self.fc2 = nn.Linear(self.l1_size, 1)
        self.act = nn.ReLU()
        self.final_act = nn.Tanh()
        self.eps = eps

    def learned_dist(self, x, y):
        x_tr = x.repeat_interleave(y.shape[0], dim=0)
        y_tr = y.repeat((x.shape[0], 1))
        xy_tr = torch.hstack((x_tr, y_tr))
        d = self.final_act(self.fc2(self.act(self.fc1(xy_tr)))).reshape(x.shape[0], y.shape[0])
        return d

    def forward(self, X, C_init):
        self.emb_dim = X.shape[1]
        self.C = C_init
        # self.d = -torch.cdist(X, C_init, p=2.0)
        self.d = -self.learned_dist(X, C_init)
        # self.a = self.softmax(self.cosine_sim(X, C_init) / self.temp)
        self.a = self.softmax(self.d / self.temp)
        self.a_sum = torch.sum(self.a, dim=0) + self.eps
        self.C_new = torch.matmul(self.a.T, X) / self.a_sum.repeat((self.emb_dim, 1)).T
        diff = torch.norm(self.C_new - self.C, p=1).item()
        i = 0
        while diff > self.threshold and i < self.max_iter:
            self.C = self.C_new
            # self.d = -torch.cdist(X, self.C, p=2.0)
            self.d = -self.learned_dist(X, C_init)
            # self.a = self.softmax(self.cosine_sim(X, self.C) / self.temp)
            self.a = self.softmax(self.d / self.temp)
            self.a_sum = torch.sum(self.a, dim=0) + self.eps
            self.C_new = torch.matmul(self.a.T, X) / self.a_sum.repeat((self.emb_dim, 1)).T
            diff = torch.norm(self.C_new - self.C, p=1).item()
            i += 1
        return self.C, self.a


class QuerySpecificDKM(nn.Module):
    def __init__(self, trans_model_name, emb_dim, device, max_len):
        super(QuerySpecificDKM, self).__init__()
        trans_model = models.Transformer(trans_model_name, max_seq_length=max_len)
        pool_model = models.Pooling(trans_model.get_word_embedding_dimension())
        dense_model = models.Dense(in_features=pool_model.get_sentence_embedding_dimension(), out_features=emb_dim,
                                   activation_function=nn.Tanh())
        self.emb_model = SentenceTransformer(modules=[trans_model, pool_model]).to(device)
        self.device = device
        self.dkm = DKM_param(emb_dim=self.emb_model.get_sentence_embedding_dimension())
        self.dkm.to(device)

    def forward(self, texts, k):
        n = len(texts)
        self.emb_vecs = torch.tensor(self.emb_model.encode(texts), device=self.device)
        init_indices = random.sample(range(n), k)
        init_c = self.emb_vecs[init_indices]
        self.C, self.a = self.dkm(self.emb_vecs, init_c)
        return self.C, self.a

    def get_clustering(self, texts, k):
        n = len(texts)
        self.emb_vecs = torch.tensor(self.emb_model.encode(texts), device=self.device)
        init_indices = random.sample(range(n), k)
        init_c = self.emb_vecs[init_indices]
        self.C, self.a = self.dkm(self.emb_vecs, init_c)
        pred_labels = torch.argmax(self.a, dim=1).detach().cpu().numpy()
        return pred_labels


class QuerySpecificClusteringModel(nn.Module):
    def __init__(self, trans_model_name, emb_dim, device, max_len):
        super(QuerySpecificClusteringModel, self).__init__()
        emb_model = models.Transformer(trans_model_name, max_seq_length=max_len)
        pool_model = models.Pooling(emb_model.get_word_embedding_dimension())
        dense_model = models.Dense(in_features=pool_model.get_sentence_embedding_dimension(), out_features=emb_dim, activation_function=nn.Tanh())
        self.qp_model = SentenceTransformer(modules=[emb_model, pool_model]).to(device)
        self.dkm = DKM()

    def forward(self, input_features, k):
        self.qp = self.qp_model(input_features)['sentence_embedding']
        init_c = self.qp[random.sample(range(self.qp.shape[0]), k)].detach().clone()
        self.C, self.a = self.dkm(self.qp, init_c)
        return self.C, self.a

    def get_embedding(self, input_features):
        self.qp = self.qp_model(input_features)['sentence_embedding']
        return self.qp

    def get_clustering(self, input_features, k):
        self.qp = self.qp_model(input_features)['sentence_embedding']
        init_c = self.qp[random.sample(range(self.qp.shape[0]), k)].detach().clone()
        self.C, self.a = self.dkm(self.qp, init_c)
        pred_labels = torch.argmax(self.a, dim=1).detach().cpu().numpy()
        return pred_labels


class SBERTTripletLossModel(nn.Module):
    def __init__(self, trans_model_name, device, max_len, triplet_margin):
        super(SBERTTripletLossModel, self).__init__()
        emb_model = models.Transformer(trans_model_name, max_seq_length=max_len)
        pool_model = models.Pooling(emb_model.get_word_embedding_dimension())
        self.emb_model = SentenceTransformer(modules=[emb_model, pool_model]).to(device)
        self.triplet_margin = triplet_margin
        self.dkm = DKM()

    def forward(self, input_features_anchor, input_features_pos, input_features_neg):
        anchor_emb = self.emb_model(input_features_anchor)['sentence_embedding']
        pos_emb = self.emb_model(input_features_pos)['sentence_embedding']
        neg_emb = self.emb_model(input_features_neg)['sentence_embedding']
        dist_pos = F.pairwise_distance(anchor_emb, pos_emb, p=2)
        dist_neg = F.pairwise_distance(anchor_emb, neg_emb, p=2)
        losses = F.relu(dist_pos - dist_neg + self.triplet_margin)
        return losses.mean()

    def get_embedding(self, input_texts):
        qp = torch.from_numpy(self.emb_model.encode(input_texts))
        return qp

    def get_clustering(self, input_texts, k):
        x = self.get_embedding(input_texts)
        init_c = x[random.sample(range(x.shape[0]), k)].detach().clone()
        self.C, self.a = self.dkm(x, init_c)
        pred_labels = torch.argmax(self.a, dim=1).detach().cpu().numpy()
        return pred_labels


class QuerySpecificClusteringModelDual(nn.Module):
    def __init__(self, q_max_len, psg_max_len, trans_model_name, emb_dim, device):
        super(QuerySpecificClusteringModelDual, self).__init__()
        q_emb_model = models.Transformer(trans_model_name, max_seq_length=q_max_len)
        q_pool_model = models.Pooling(q_emb_model.get_word_embedding_dimension())
        q_dense_model = models.Dense(in_features=q_pool_model.get_sentence_embedding_dimension(), out_features=emb_dim,
                                     activation_function=nn.Sigmoid())
        self.q_model = SentenceTransformer(modules=[q_emb_model, q_pool_model, q_dense_model]).to(device)
        emb_model = models.Transformer(trans_model_name, max_seq_length=psg_max_len)
        pool_model = models.Pooling(emb_model.get_word_embedding_dimension())
        dense_model = models.Dense(in_features=pool_model.get_sentence_embedding_dimension(), out_features=emb_dim,
                                     activation_function=nn.Tanh())
        self.p_model = SentenceTransformer(modules=[emb_model, pool_model, dense_model]).to(device)
        self.dkm = DKM()

    def forward(self, q_features, psg_features, k):
        self.qw = self.q_model(q_features)['sentence_embedding']
        self.p = self.p_model(psg_features)['sentence_embedding']
        self.qp = self.qw * self.p
        self.C, self.a = self.dkm(self.qp, self.qp[:k])
        self.sim_mat = 1 / (1 + torch.cdist(self.a, self.a))
        return self.sim_mat