import os, random, torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.preprocessing import scale
import numpy as np
import pandas as pd
from torch.nn import Parameter

import warnings
warnings.filterwarnings("ignore")

def functional_linear3d(input, weight, bias=None):
    r"""
    Apply a linear transformation to the incoming data: :math:`y = xA^T + b`.

    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    output = input.transpose(0, 1).matmul(weight)
    if bias is not None:
        output += bias.unsqueeze(1)
    return output.transpose(0, 1)

class Linear3D(torch.nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
        Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            `(out_features x in_features)`
        bias:   the learnable bias of the module of shape `(out_features)`
    """

    def __init__(self, channels, in_features, out_features, batch_size=-1, bias=True, noise=False):
        super(Linear3D, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.channels = channels
        if noise:
            self.in_features += 1
        self.weight = Parameter(torch.Tensor(channels, self.in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(channels, out_features))
        else:
            self.register_parameter('bias', None)
        if noise:
            self.register_buffer("noise", torch.Tensor(batch_size, channels, 1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj_matrix=None, permutation_matrix=None):
        input_ = [input]

        if input.dim() == 2:
            if permutation_matrix is not None:
                input_.append(input.unsqueeze(1).expand([input.shape[0], self.channels, permutation_matrix.shape[1]]))
            elif hasattr(self, "noise"):
                input_.append(input.unsqueeze(1).expand([input.shape[0], self.channels, self.in_features - 1]))
            else:
                input_.append(input.unsqueeze(1).expand([input.shape[0], self.channels, self.in_features]))

        if adj_matrix is not None and permutation_matrix is not None:
            input_.append(
                (input_[-1].transpose(0, 1) @ (adj_matrix.t().unsqueeze(2) * permutation_matrix)).transpose(0, 1))
        elif adj_matrix is not None:
            input_.append(input_[-1] * adj_matrix.t().unsqueeze(0))
        elif permutation_matrix is not None:
            input_.append((input_[-1].transpose(0, 1) @ permutation_matrix).t())

        if hasattr(self, 'noise'):
            self.noise.normal_()
            input_.append(torch.cat([input_[-1], self.noise], 2))

        return functional_linear3d(input_[-1], self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def apply_filter(self, permutation_matrix):
        transpose_weight = self.weight.transpose(1, 2) @ permutation_matrix
        self.weight = Parameter(transpose_weight.transpose(1, 2))

class CustomizedRelaModule(torch.nn.Module):
    """Ensemble of all the generators."""

    def permutation_matrix(self, skeleton, data_shape, max_dim):
        # skeleton 90 * 90
        # nb_vars: number of variables 90
        # max_dim == 90 -1 == 89

        reshape_skeleton = torch.zeros(self.nb_vars, int(data_shape[1]), max_dim)
        for channel in range(self.nb_vars):
            perm_matrix = skeleton[:, channel] * torch.eye(data_shape[1], data_shape[1])
            skeleton_list = [i for i in torch.unbind(perm_matrix, 1) if np.count_nonzero(i.numpy()) > 0]
            perm_matrix = torch.stack(skeleton_list, 1) if len(skeleton_list) > 0 else torch.zeros(data_shape[1], 1)
            reshape_skeleton[channel, :, :perm_matrix.shape[1]] = perm_matrix
        return reshape_skeleton

    def __init__(self, data_shape, nh, output_dim=1):
        """
           Init the model.
           data_shape = 135 * 90  = (the length of time points * brain regions)
           nh(int): number of hidden units = 100
           skeleton:  causal structure of brain network  90 * 90
        """
        super(CustomizedRelaModule, self).__init__()
        layers = []

        self.nb_vars = data_shape[1]  # 90
        cat_reshape = torch.eye(self.nb_vars, self.nb_vars)

        skeleton = torch.eye(self.nb_vars)
        skeleton = 1 - skeleton

        # Re-dimension the skeleton according to the categorical vars
        skeleton = cat_reshape.t() @ skeleton @ cat_reshape
        max_dim = torch.as_tensor(skeleton.sum(dim=0).max(), dtype=torch.int)

        reshape_skeleton = self.permutation_matrix(skeleton, data_shape, max_dim)

        self.input_layer = Linear3D(self.nb_vars, max_dim, nh)
        layers.append(torch.nn.Tanh())

        self.layers = torch.nn.Sequential(*layers)
        self.output_layer = Linear3D(self.nb_vars, nh, output_dim)

        self.register_buffer('skeleton', reshape_skeleton)
        self.register_buffer("categorical_matrix", cat_reshape)

        self.adj_matrix = torch.nn.Parameter(torch.FloatTensor(self.nb_vars, self.nb_vars))
        self.adj_matrix.data.normal_()
        self.neurons = torch.nn.Parameter(torch.FloatTensor(nh, self.nb_vars))
        self.neurons.data.normal_()

    def forward(self, data):
        """
        Forward through all the generators.
           data---(i.e., the preprocessed time series data X):  134 * 116
           adj_matrix (save the estimation result of causal effect): 116 * 116
        """
        input = self.input_layer(data, self.categorical_matrix.t() @ self.adj_matrix, self.skeleton)
        branch = self.layers(input)

        output = self.output_layer(branch, self.neurons)  # 135 * 90 * 1
        return self.adj_matrix, output.squeeze(2)

    def reset_parameters(self):
        self.output_layer.reset_parameters()
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.input_layer.reset_parameters()

class DirectedGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(DirectedGraphConvolution, self).__init__()
        self.weight_in = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight_out = nn.Parameter(torch.Tensor(in_features, out_features))

    def forward(self, x, adj):
        '''
        :param x: n * feature
        :param adj:  n * n
        :return: node embedding out  batch * nodeNum * out_dim
        '''
        x_in = torch.matmul(adj, torch.matmul(x, self.weight_in))
        x_out = torch.matmul(torch.transpose(adj, 0, 1), torch.matmul(x, self.weight_out))

        output = (x_in + x_out)/2
        return output

class TopKPool(nn.Module):
    def __init__(self, num_feature, k):
        super(TopKPool, self).__init__()
        self.k = k
        self.w = nn.Parameter(torch.tensor(num_feature).float())

    def forward(self, H):
        sub_nodes_score = torch.zeros(size=(H.shape[0], H.shape[1]))

        for subNum in range(H.shape[0]):
            sub_nodes_score[subNum, :] = torch.sum(H[subNum, :, :] * self.w, dim=1) / torch.norm(self.w, p=2)

        score = torch.sum(sub_nodes_score, dim=0)
        _, index = torch.topk(score, self.k)

        sorted_index = torch.argsort(index)
        return index[sorted_index]

class CRGNN(nn.Module):
    def __init__(self, num_nodes, hidden_units):
        """
        Args:
            num_nodes: int
                The number of variables (N)
            hidden_units: int
                Number of hidden units per layer
            hidden_layers: int
                Number of hidden layers
        """
        super(CRGNN, self).__init__()

        self.num_nodes = num_nodes
        self.hidden_units = hidden_units

        self.gsl = CustomizedRelaModule(data_shape=(featureLen, self.num_nodes), nh=self.hidden_units)

        self.dgcn1 = DirectedGraphConvolution(in_features=featureLen, out_features=32)
        self.pool1 = TopKPool(num_feature=32, k=30)

        self.dgcn2 = DirectedGraphConvolution(in_features=32, out_features=8)
        self.pool2 = TopKPool(num_feature=32, k=10)

        self.mlp = nn.Sequential(
            nn.Linear(10 * 8, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, time_input):
        batch_sub, time_points, number_of_nodes = time_input.shape

        predictions = torch.zeros(size=(batch_sub, time_points, number_of_nodes)).to(device)
        graphs = torch.zeros((batch_sub, self.num_nodes, self.num_nodes)).to(device)

        node_features = time_input.permute(0, 2, 1)

        regMatrix = (1 - torch.zeros(size=(NumNodes, NumNodes))).to(device)

        for numSub in range(batch_sub):
            graph, time_pred = self.gsl(data=time_input[numSub, :, :])
            predictions[numSub, :, :] = time_pred
            graphs[numSub, :, :] = graph * regMatrix

        H1 = torch.zeros(size=(batch_sub, self.num_nodes, 32)).to(device)

        # DGCN1
        for batch in range(batch_sub):
            H1[batch, :, :] = self.dgcn1(x=node_features[batch, :, :], adj=graphs[batch, :, :])

        # TopK Pooling1
        retain_index1 = self.pool1(H=H1).to(device)

        H1 = H1[:, retain_index1, :]                 # batch * 30 * 32
        A1 = torch.index_select(torch.index_select(graphs, dim=1, index=retain_index1), dim=2, index=retain_index1)

        H2 = torch.zeros(size=(batch_sub, 30, 8)).to(device)

        # DGCN2
        for batch in range(batch_sub):
            H2[batch, :, :] = self.dgcn2(x=H1[batch, :, :], adj=A1[batch, :, :])
        # TopK Pooling2
        retain_index2 = self.pool2(H=H2)
        H2_retain = H2[:, retain_index2, :]

        features = torch.reshape(H2_retain, shape=(-1, 10 * 8))

        out = F.softmax(self.mlp(features), dim=-1)
        return predictions, graphs, out, retain_index1, retain_index2

class TSDataset(torch.utils.data.Dataset):
    def __init__(self, dataPathList, labels):
        self.dataPathList = dataPathList
        self.labels = labels

    def __len__(self):
        return len(self.dataPathList)

    def __getitem__(self, index):
        data = np.loadtxt(self.dataPathList[index], dtype=np.float32)[:featureLen, :90]
        np.nan_to_num(data, nan=0)

        data = data.astype('float32')
        data = torch.from_numpy(data)

        X = data
        label = np.array(self.labels[index])
        return X, label

if __name__ == '__main__':
    dataPath = '/data/xiazhengwang/ABIDEI_TimeSeries'

    interResultPath = '/data/xiazhengwang/CRGNN/interResultPath_ABIDE/'

    lambda1 = 0.0001
    lambda2 = 0.0001
    NumNodes = 90
    EPOCHS = 50
    BATCH_SIZE = 16
    LR = 0.001
    featureLen = 78

    data_path = []
    data_label = []

    for site in os.listdir(dataPath):
        sitePath = dataPath + '/' + site
        for cate in os.listdir(sitePath):
            catePath = sitePath + '/' + cate
            for tFile in os.listdir(catePath):
                filePath = catePath + '/' + tFile
                data_path.append(filePath)
                if cate == 'Control':
                    data_label.append(0)
                else:
                    data_label.append(1)

    combined = list(zip(data_path, data_label))
    random.shuffle(combined)
    shuffled_data, shuffled_labels = zip(*combined)

    # 10-fold CV
    CV = 10
    splitNum = int(len(shuffled_data)/CV)

    testResultCSV = pd.DataFrame(columns=['Name', 'TrueLabel', 'Prediction'])

    testNum = 0
    for num in range(CV):
        if num != CV - 1:
            test_path = shuffled_data[num * splitNum: (num + 1) * splitNum]
            test_label = shuffled_labels[num * splitNum: (num + 1) * splitNum]

            train_path = shuffled_data[0: num * splitNum] + shuffled_data[(num+1) * splitNum:]
            train_label = shuffled_labels[0: num * splitNum] + shuffled_labels[(num + 1) * splitNum:]
        else:
            test_path = shuffled_data[num * splitNum:]
            test_label = shuffled_labels[num * splitNum:]

            train_path = shuffled_data[0: num * splitNum]
            train_label = shuffled_labels[0: num * splitNum]

        # read the dataset
        train_dataset = TSDataset(train_path, train_label)
        test_dataset = TSDataset(test_path, test_label)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model = CRGNN(num_nodes=90, hidden_units=100)
        model.to(device)

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params=params, lr=LR, weight_decay=0.0001)

        testFileName = []
        for name in test_path:
            testFileName.append(name.split('/')[-2] + '/' + name.split('/')[-1])

        # weight = torch.tensor([2.0, 1.0]).to(device)

        # train the model
        for epo in range(EPOCHS):
            train_loss = 0
            train_correct = 0.0
            model.train()
            print("Epoch {}".format(epo), '------')

            num_train = 0

            if not os.path.exists(interResultPath + 'Epoch' + str(epo)):
                os.makedirs(interResultPath + 'Epoch' + str(epo) + '/NC')
                os.makedirs(interResultPath + 'Epoch' + str(epo) + '/MCI')

            for _, batch_train_data in enumerate(train_loader):
                input4gsl, labels = batch_train_data
                input4gsl, labels = Variable(input4gsl.to(device)), Variable(labels.to(device).long())

                predictions, graphs, out, retain_index1, retain_index2 = model(input4gsl)
                pred = out.data.max(1, keepdim=True)[1]

                train_correct += pred.eq(labels.data.view_as(pred)).cpu().sum()

                # cross_entropy = nn.CrossEntropyLoss(weight=weight)
                cross_entropy = nn.CrossEntropyLoss()
                CE = cross_entropy(out, labels)

                MSE = nn.MSELoss(reduction='mean')

                loss_pred = []
                graph_L1 = []

                regMatrix = (1 - torch.zeros(size=(NumNodes, NumNodes))).to(device)

                for subNum in range(predictions.shape[0]):
                    dis_pred = predictions[subNum, :, :]
                    loss_pred.append(MSE(dis_pred, input4gsl[subNum, :, :]))

                    graph = graphs[subNum, :, :] * regMatrix

                    if labels[subNum] == 0:
                        np.savetxt(interResultPath + 'Epoch' + str(epo) + '/NC/' + str(num_train) + '.txt', graph.detach().cpu().numpy())
                        num_train = num_train + 1
                    else:
                        np.savetxt(interResultPath + 'Epoch' + str(epo) + '/MCI/' + str(num_train) + '.txt', graph.detach().cpu().numpy())
                        num_train = num_train + 1

                    graph_L1.append(torch.norm(graph, p=1))

                np.savetxt(interResultPath + 'Epoch' + str(epo) + '/index1.txt', retain_index1.detach().cpu().numpy())
                np.savetxt(interResultPath + 'Epoch' + str(epo) + '/index2.txt', retain_index2.detach().cpu().numpy())

                loss_pred = torch.mean(torch.Tensor(loss_pred))
                loss_graph_L1 = torch.mean(torch.Tensor(graph_L1))

                loss = CE + lambda1 * loss_pred + lambda2 * loss_graph_L1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                print("Training: CE loss: {:.3f}, Predict loss: {:.3f}, L1 loss: {:.3f}.".format(CE.item(), loss_pred.item(), loss_graph_L1.item()))

            # LR = LR * (0.9999 ** epo)
            train_accuracy = float(train_correct) / len(train_dataset)

            model.eval()
            test_loss = 0
            test_correct = 0.0

            predictions_list = []
            for _, batch_test_data in enumerate(test_loader):
                input4gsl, labels = batch_test_data
                input4gsl, labels = Variable(input4gsl.to(device)), Variable(labels.to(device).long())

                predictions, graphs, out, retain_index1, retain_index2 = model(input4gsl)

                pred = out.data.max(1, keepdim=True)[1]
                test_correct += pred.eq(labels.data.view_as(pred)).cpu().sum()

                for temp in range(pred.shape[0]):
                    predictions_list.append(int(pred.cpu()[temp]))

                # cross_entropy = nn.CrossEntropyLoss(weight=weight)
                cross_entropy = nn.CrossEntropyLoss()
                CE = cross_entropy(out, labels)

                MSE = nn.MSELoss(reduction='mean')

                loss_pred = []
                graph_L1 = []

                regMatrix = (1 - torch.zeros(size=(NumNodes, NumNodes))).to(device)

                for subNum in range(predictions.shape[0]):
                    dis_pred = predictions[subNum, :, :]
                    loss_pred.append(MSE(dis_pred, input4gsl[subNum, :, :]))

                    graph = graphs[subNum, :, :] * regMatrix
                    graph_L1.append(torch.norm(graph, p=1))

                loss_pred = torch.mean(torch.Tensor(loss_pred))
                loss_graph_L1 = torch.mean(torch.Tensor(graph_L1))

                test_loss = CE + lambda1 * loss_pred + lambda2 * loss_graph_L1

                print("Testing: CE loss: {:.3f}, Predict loss: {:.3f}, L1 loss: {:.3f}".format(CE.item(), loss_pred.item(), loss_graph_L1.item()))

            test_accuracy = float(test_correct) / len(test_dataset)

            print('Testing: CV: {}, Epoch: {}, Train loss: {:.3f}, Train accuracy: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.3f}'.format(
                    num, epo, train_loss / len(train_dataset), train_accuracy, test_loss / len(test_dataset), test_accuracy))

        for num_test_cv in range(len(test_label)):
            rowResult = {'Name': testFileName[num_test_cv], 'TrueLabel': test_label[num_test_cv], 'Prediction': predictions_list[num_test_cv]}
            testResultCSV.loc[len(testResultCSV)] = rowResult

    testResultCSV.to_csv('ABIDE.csv', sep=',', index=False, header=True)