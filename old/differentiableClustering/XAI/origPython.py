
"""
However, we further notice that when W and b are decoupled, directly optimizing them would lead to divergent and unstable training. To address this issue,
we propose normalizing both the cluster layer weight and its gradient to achieve a stable training,
In practice, we experimentally normalize the gradient to 10% of
the length of Wj .

"""
def inference():
    net.compute_cluster_center(alpha) # based on scaled cluster weight
    net.eval()# as far as I get just internal function of the network
    feature_vector = []
    labels_vector = []
    pred_vector = []
    with torch.no_grad():
        for step, (x, y) in enumerate(data_loader_test):
            x = x.cuda()
            with torch.no_grad():
                z = net.encode(x)
                pred = net.predict(z)
            #extent just add vector to the end of the other vector ...
            feature_vector.extend(z.detach().cpu().numpy())
            labels_vector.extend(y.numpy())
            pred_vector.extend(pred.detach().cpu().numpy())
    return feature_vector, labels_vector, pred_vector

    #basically typical encoder decoder architecture with convolutions and transposed convolutions
    net = NetConv(channel=1, inner_dim=784, class_num=class_num).cuda()

    optimizer = torch.optim.Adadelta(net.parameters())
    criterion = nn.MSELoss(reduction="mean")
    net.normalize_cluster_center(alpha)

    """
    below we see 
    1) encoding data in z - getting reduced representation
    2) we prepare cluster batch by applying linear to this reduced representation
        looking at https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        critical seem to be fact that only last dimension in linear increases - so it seems it is in this point doing sth like adding as many channels 
        as there are classes - Hovewer at this point it seems that it can be done iteratively to keep all in memory
    3) we get soft labels by soft max and hard ones by argmax of this linear layer that has 

    4) lines below seem to be crucial contribution of this paper as it give the clustering loss
        alpha - seem to be hyperparameter
        delta is tensor with argmaxed labels - just reshaped
            loss_clu_batch = 2 * alpha - torch.mul(delta, cluster_batch)
            loss_clu_batch = 0.01 / alpha * loss_clu_batch.mean()
        so we basically scale and normalize the labels
    5) loss_rec -     loss related to encoder decoder scheme

    6)lines below are also criticall important as here the normalization of gradients is performed to avoid 
        the trivial solution
                if epoch % 2 == 0:
                net.cluster_layer.weight.grad = (
                    F.normalize(net.cluster_layer.weight.grad, dim=1) * 0.2 * alpha
                )
       normalize_cluster_center - is also normalizing gradients         
    """    
    for epoch in range(start_epoch, epochs):
        loss_clu_epoch = loss_rec_epoch = 0
        net.train()
        for step, (x, y) in enumerate(data_loader):
            #from encoder framework
            z = net.encode(x)

            if epoch % 2 == 1:
                cluster_batch = net.cluster(z)
            else:
                cluster_batch = net.cluster(z.detach()) #cluster is just linear ...
            soft_label = F.softmax(cluster_batch.detach(), dim=1)
            hard_label = torch.argmax(soft_label, dim=1)
            delta = torch.zeros((batch_size, 10), requires_grad=False).cuda()
            for i in range(batch_size):
                delta[i, torch.argmax(soft_label[i, :])] = 1
            loss_clu_batch = 2 * alpha - torch.mul(delta, cluster_batch)
            loss_clu_batch = 0.01 / alpha * loss_clu_batch.mean()

            x_ = net.decode(z)
            loss_rec = criterion(x, x_)

            loss = loss_rec + loss_clu_batch
            optimizer.zero_grad()
            loss.backward()
            if epoch % 2 == 0:
                net.cluster_layer.weight.grad = (
                    F.normalize(net.cluster_layer.weight.grad, dim=1) * 0.2 * alpha
                )
            else:
                net.cluster_layer.zero_grad()
            optimizer.step()
            net.normalize_cluster_center(alpha)
            loss_clu_epoch += loss_clu_batch.item()
            loss_rec_epoch += loss_rec.item()
        print(
            f"Epoch [{epoch}/{epochs}]\t Clu Loss: {loss_clu_epoch / len(data_loader)}\t Rec Loss: {loss_rec_epoch / len(data_loader)}"
        )


import math

import torch
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, dim, class_num):
        super(Net, self).__init__()
        self.class_num = class_num
        self.encoder = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(dim, 500, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(500, 500, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(500, 2000, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(2000, 10, bias=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 2000, bias=True),
            nn.ReLU(),
            nn.Linear(2000, 500, bias=True),
            nn.ReLU(),
            nn.Linear(500, 500, bias=True),
            nn.ReLU(),
            nn.Linear(500, dim, bias=True),
            nn.Sigmoid(),
        )
        self.cluster_layer = nn.Linear(10, class_num, bias=False)
        self.cluster_center = torch.rand([class_num, 10], requires_grad=False)

    def encode(self, x):
        x = self.encoder(x)
        x = F.normalize(x)
        return x

    def decode(self, x):
        return self.decoder(x)

    def cluster(self, z):
        return self.cluster_layer(z) #just a linear ... that as far as I get it get from output to set number of classes it has in =10 as this is the dim of net output

    def init_cluster_layer(self, alpha, cluster_center):
        self.cluster_layer.weight.data = 2 * alpha * cluster_center

    def compute_cluster_center(self, alpha):
        self.cluster_center = 1.0 / (2 * alpha) * self.cluster_layer.weight
        return self.cluster_center

    def normalize_cluster_center(self, alpha):
        self.cluster_layer.weight.data = (
            F.normalize(self.cluster_layer.weight.data, dim=1) * 2.0 * alpha
        )

    def predict(self, z):
        distance = torch.cdist(z, self.cluster_center, p=2)
        prediction = torch.argmin(distance, dim=1)
        return prediction

    def set_cluster_centroid(self, mu, cluster_id, alpha):
        self.cluster_layer.weight.data[cluster_id] = 2 * alpha * mu


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.size(0),) + self.shape)


class NetConv(Net):
    def __init__(self, channel, inner_dim, class_num):
        super(NetConv, self).__init__(dim=inner_dim, class_num=class_num)
        self.class_num = class_num
        self.inner_dim = inner_dim
        self.kernel_size = int(math.sqrt(inner_dim / 16))
        self.encoder = nn.Sequential(
            nn.Conv2d(channel, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(inner_dim, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 10, bias=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, inner_dim, bias=True),
            Reshape(16, self.kernel_size, self.kernel_size),
            nn.ConvTranspose2d(
                16, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, channel, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )



def evaluate(label, pred):
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    # f = metrics.fowlkes_mallows_score(label, pred)
    pred_adjusted = get_y_preds(label, pred, len(set(label)))
    acc = metrics.accuracy_score(pred_adjusted, label)
    return nmi, ari, acc


def evaluate_others(label, pred):
    ami = metrics.adjusted_mutual_info_score(label, pred)
    homo, comp, v_mea = metrics.homogeneity_completeness_v_measure(label, pred)
    return ami, homo, comp, v_mea


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels


def get_y_preds(y_true, cluster_assignments, n_clusters):
    """
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)
    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset
    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    """
    confusion_matrix = metrics.confusion_matrix(
        y_true, cluster_assignments, labels=None
    )
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred                    