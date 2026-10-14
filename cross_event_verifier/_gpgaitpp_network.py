"""Inference primitives adapted from BNU-IVC/FastPoseGait.

Pinned source and per-file hashes: models/gpgaitpp_source.json.
No training, dataset, distributed launcher or evaluator is included.
The only operator change is device-aware placement instead of CUDA-only placement.
Upstream terms are recorded in THIRD_PARTY_NOTICES.md.
"""
import math
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

def clones(module, count):
    return nn.ModuleList(copy.deepcopy(module) for _ in range(count))

def is_list_or_tuple(value):
    return isinstance(value, (list, tuple))

class Graph():
    """
    # Thanks to YAN Sijie for the released code on Github (https://github.com/yysijie/st-gcn)
    """
    def __init__(self, joint_format='coco', max_hop=2, dilation=1):
        self.joint_format = joint_format
        self.max_hop = max_hop
        self.dilation = dilation

        # get edges
        self.num_node, self.edge, self.connect_joint = self._get_edge()

        # get adjacency matrix
        self.A = self._get_adjacency()

    def __str__(self):
        return self.A

    def _get_edge(self):
        if self.joint_format == 'coco':
            # keypoints = {
            #     0: "nose",
            #     1: "left_eye",
            #     2: "right_eye",
            #     3: "left_ear",
            #     4: "right_ear",
            #     5: "left_shoulder",
            #     6: "right_shoulder",
            #     7: "left_elbow",
            #     8: "right_elbow",
            #     9: "left_wrist",
            #     10: "right_wrist",
            #     11: "left_hip",
            #     12: "right_hip",
            #     13: "left_knee",
            #     14: "right_knee",
            #     15: "left_ankle",
            #     16: "right_ankle"
            # }
            num_node = 17
            self_link = [(i, i) for i in range(num_node)]
            neighbor_link = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6),
                             (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
                             (11, 13), (13, 15), (12, 14), (14, 16)]
            self.edge = self_link + neighbor_link
            self.center = 0
            self.flip_idx = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
            connect_joint = np.array([5,0,0,1,2,0,0,5,6,7,8,5,6,11,12,13,14])

        elif self.joint_format == 'coco-no-head':
            num_node = 12
            self_link = [(i, i) for i in range(num_node)]
            neighbor_link = [(0, 1),
                             (0, 2), (2, 4), (1, 3), (3, 5), (0, 6), (1, 7), (6, 7),
                             (6, 8), (8, 10), (7, 9), (9, 11)]
            self.edge = self_link + neighbor_link
            self.center = 0
            connect_joint = np.array([3,1,0,2,4,0,6,8,10,7,9,11])

        elif self.joint_format =='alphapose' or self.joint_format =='openpose':
            num_node = 18
            self_link = [(i, i) for i in range(num_node)]
            neighbor_link = [(0, 1), (0, 14), (0, 15), (14, 16), (15, 17),
                             (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
                             (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13)]
            self.edge = self_link + neighbor_link
            self.center = 1
            self.flip_idx = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16]
            connect_joint = np.array([1,1,1,2,3,1,5,6,2,8,9,5,11,12,0,0,14,15])

        else:
            num_node, neighbor_link, connect_joint = 0, [], []
            raise ValueError('Error: Do NOT exist this dataset: {}!'.format(self.joint_format))
        self_link = [(i, i) for i in range(num_node)]
        edge = self_link + neighbor_link
        return num_node, edge, connect_joint

    def _get_hop_distance(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_adjacency(self):
        hop_dis = self._get_hop_distance()
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[hop_dis == hop] = 1
        normalize_adjacency = self._normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]
        return A

    def _normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD

class GraphPartition():
    """
    Support for the joint format of coco17.
    """
    def __init__(self, joint_format='coco'):

        if joint_format == 'coco':
            whole_body = [list(range(0,17))]
            self.partition_list = []
            parts5 = [
                np.array([5, 7, 9]),         # left_arm
                np.array([6, 8, 10]),        # right_arm
                np.array([11, 13, 15]),      # left_leg
                np.array([12, 14, 16]),      # right_leg
                np.array([0, 1, 2, 3, 4]),   # head
            ]
            parts2_1 = [
                np.array([5, 7, 9, 6, 8, 10, 0, 1, 2, 3, 4]), # top_half
                np.array([11, 13, 15, 12, 14, 16])            # bottom_half
            ]
            parts2_2 = [
                np.array([5, 7, 9, 6, 8, 10]),                    # arms
                np.array([11, 13, 15, 12, 14, 16, 0, 1, 2, 3, 4]) # head + legs
            ]
            parts3_1 = [
                np.array([5, 7, 9, 6, 8, 10]),      # arms
                np.array([11, 13, 15, 12, 14, 16]), # legs
                np.array([0, 1, 2, 3, 4])           # head
            ]
            parts3_2 = [
                np.array([5, 7, 9, 12, 14, 16]),  # left_arm + right_leg
                np.array([6, 8, 10, 11, 13, 15]), # right_arm + left_leg
                np.array([0, 1, 2, 3, 4])         # head
            ]
            self.partition_list.append(whole_body)
            self.partition_list.append(parts5)
            self.partition_list.append(parts2_1)
            self.partition_list.append(parts2_2)
            self.partition_list.append(parts3_1)
            self.partition_list.append(parts3_2)

        elif joint_format == 'coco-no-head':
            self.partition_list = []
            whole_body = [list(range(0,12))]
            parts4 = [
                np.array([0, 2, 4]),     # left_arm
                np.array([1, 3, 5]),     # right_arm
                np.array([6, 8, 10]),    # left_leg
                np.array([7, 9, 11]),    # right_leg
            ]
            parts2_1 = [
                np.array([0, 2, 4, 1, 3, 5]),  # top_half
                np.array([6, 8, 10, 7, 9, 11]) # bottom_half
            ]
            parts2_2 = [
                np.array([0, 2, 4, 7, 9, 11]),  # left_arm + right_leg
                np.array([1, 3, 5, 6, 8, 10]),  # right_arm + left_leg
            ]
            self.partition_list.append(whole_body)
            self.partition_list.append(parts4)
            self.partition_list.append(parts2_1)
            self.partition_list.append(parts2_2)

    def __call__(self, part=None):
        if part:
            if not isinstance(part, int):
                raise ValueError('Error: give a integer for partition!')
            ret_partition = self.partition_list[part]
        else:
            ret_partition = self.partition_list[0]
        return ret_partition

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x * (torch.tanh(F.softplus(x)))

class UnitConv2D(nn.Module):
    '''
    This class is used in GaitTR[TCN_ST] block.
    '''

    def __init__(self, D_in, D_out, kernel_size=9, stride=1, dropout=0.1, bias=True):
        super(UnitConv2D,self).__init__()
        pad = int((kernel_size-1)/2)
        self.conv = nn.Conv2d(D_in,D_out,kernel_size=(kernel_size,1)
                            ,padding=(pad,0),stride=(stride,1),bias=bias)
        self.bn = nn.BatchNorm2d(D_out)
        self.relu = Mish()
        self.dropout = nn.Dropout(dropout, inplace=False)
        #initalize
        self.conv_init(self.conv)

    def forward(self,x):
        x = self.dropout(x)
        x = self.bn(self.relu(self.conv(x)))
        return x

    def conv_init(self,module):
        n = module.out_channels
        for k in module.kernel_size:
            n = n*k
        module.weight.data.normal_(0, math.sqrt(2. / n))

class PAGCN_Plus(nn.Module):
    '''
    This class implements Part-Aware Graph Convolution used in GPGait++
    '''
    def __init__(self, in_channels, out_channels, A, joint_format, coff_embedding=4, num_subset=3):
        super(PAGCN_Plus, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(A)
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(A, requires_grad=False)
        self.partition = GraphPartition(joint_format=joint_format)
        self.num_subset = num_subset
        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = Mish()

    def Partition2Mask(self, partition, num_point):
        part_mask = torch.zeros([num_point, num_point],dtype=torch.float)
        for part in partition:
            for i in part:
                for j in part:
                    part_mask[i, j] = 1.
        part_mask = part_mask.unsqueeze(0)
        return part_mask

    def forward(self, x, A, part=None):
        N, C, T, V = x.size()
        A = A.to(device=x.device, dtype=x.dtype)
        if part:
            partition = self.partition(part)
            num_point = A.shape[-1]
            part_mask = self.Partition2Mask(partition, num_point)
            part_mask = part_mask.to(device=x.device, dtype=x.dtype)
        A = A + self.PA
        y = None
        for i in range(self.num_subset):
            q = self.conv_a[i](x).permute(0, 2, 3, 1).contiguous().view(N * T, V, self.inter_c) # x[N,C,T,V]->[N,T,V,C]
            k = self.conv_b[i](x).permute(0, 2, 1, 3).contiguous().view(N * T, self.inter_c, V) # x[N,C,T,V]->[N,T,C,V]
            CA = self.soft(torch.matmul(q, k) / q.size(-1))  # N*T V V
            A1 = CA + A[i]
            if part:
                A1 = A1 * part_mask
            v = x.permute(0, 2, 1, 3).contiguous().view(N * T, C, V)
            z = self.conv_d[i](torch.matmul(v, A1).view(N, T, C, V).permute(0, 2, 1, 3).contiguous())
            y = z + y if y is not None else z

        y = self.bn(y)
        return self.relu(y)

class Part_AGCN_TCN(nn.Module):
    def __init__(self, in_channels, out_channels, A, joint_format='coco', mask=True):
        super(Part_AGCN_TCN,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.A = A
        self.mask=mask
        out_asg = out_channels // 2
        out_res = out_channels - out_asg
        out_tcn = out_channels // 3 + 1
        out_res_s = out_channels // 3
        out_down = out_channels - out_res_s - out_tcn
        self.asg = PAGCN_Plus(in_channels=self.in_channels,out_channels=out_asg,A=self.A,joint_format=joint_format)
        self.tcn = UnitConv2D(D_in=self.out_channels, D_out=out_tcn, kernel_size=9)

        self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_res, 1),
                nn.BatchNorm2d(out_res),
            )
        self.residual_s = nn.Sequential(
                nn.Conv2d(self.out_channels, out_res_s, 1),
                nn.BatchNorm2d(out_res_s),
            )
        self.down = UnitConv2D(D_in=self.in_channels, D_out=out_down, kernel_size=1, dropout=0)

    def forward(self, x, A, part=None):
        if not self.mask:
            part = None
        x0 = torch.cat([self.asg(x, A, part), self.residual(x)], dim=1)
        y = torch.cat([self.tcn(x0), self.residual_s(x0), self.down(x)], dim=1)
        return y

class PackSequenceWrapper(nn.Module):
    '''
    This class implements Set Pooling.
    '''
    def __init__(self, pooling_func):
        super(PackSequenceWrapper, self).__init__()
        self.pooling_func = pooling_func

    def forward(self, seqs, seqL, dim=2, options={}):
        """
            In  seqs: [n, c, s, ...]
            Out rets: [n, ...]
        """
        if seqL is None:
            return self.pooling_func(seqs, **options)
        seqL = seqL[0].data.cpu().numpy().tolist()
        start = [0] + np.cumsum(seqL).tolist()[:-1]

        rets = []
        for curr_start, curr_seqL in zip(start, seqL):
            narrowed_seq = seqs.narrow(dim, curr_start, curr_seqL)
            rets.append(self.pooling_func(narrowed_seq, **options))
        if len(rets) > 0 and is_list_or_tuple(rets[0]):
            return [torch.cat([ret[j] for ret in rets])
                    for j in range(len(rets[0]))]
        return torch.cat(rets)

class SemanticPyramidPooling():
    '''
    This class implements Semantic Pyramid Pooling
    '''
    def __init__(self, joint_format='coco'):
        self.partition = GraphPartition(joint_format=joint_format)

    def __call__(self, x, part=None):
        feature = []
        partition = self.partition(part)
        for part in partition:
            temp = x[...,part].mean(-1) + x[...,part].max(-1)[0]
            temp = temp.unsqueeze(-1)
            feature.append(temp)

        return torch.cat(feature,-1)

class SeparateFCs(nn.Module):
    def __init__(self, parts_num, in_channels, out_channels, norm=False):
        super(SeparateFCs, self).__init__()
        self.p = parts_num
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, out_channels)))
        self.norm = norm

    def forward(self, x):
        """
            x: [n, c_in, p]
            out: [n, c_out, p]
        """
        x = x.permute(2, 0, 1).contiguous()
        if self.norm:
            out = x.matmul(F.normalize(self.fc_bin, dim=1))
        else:
            out = x.matmul(self.fc_bin)
        return out.permute(1, 2, 0).contiguous()

class SeparateBNNecks(nn.Module):
    """
        GaitSet: Bag of Tricks and a Strong Baseline for Deep Person Re-Identification
        CVPR Workshop:  https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf
        Github: https://github.com/michuanhaohao/reid-strong-baseline
    """

    def __init__(self, parts_num, in_channels, class_num, norm=True, parallel_BN1d=True):
        super(SeparateBNNecks, self).__init__()
        self.p = parts_num
        self.class_num = class_num
        self.norm = norm
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, class_num)))
        if parallel_BN1d:
            self.bn1d = nn.BatchNorm1d(in_channels * parts_num)
        else:
            self.bn1d = clones(nn.BatchNorm1d(in_channels), parts_num)
        self.parallel_BN1d = parallel_BN1d

    def forward(self, x):
        """
            x: [n, c, p]
        """
        if self.parallel_BN1d:
            n, c, p = x.size()
            x = x.view(n, -1)  # [n, c*p]
            x = self.bn1d(x)
            x = x.view(n, c, p)
        else:
            x = torch.cat([bn(_x) for _x, bn in zip(
                x.split(1, 2), self.bn1d)], 2)  # [p, n, c]
        feature = x.permute(2, 0, 1).contiguous()
        if self.norm:
            feature = F.normalize(feature, dim=-1)  # [p, n, c]
            logits = feature.matmul(F.normalize(
                self.fc_bin, dim=1))  # [p, n, c]
        else:
            logits = feature.matmul(self.fc_bin)
        return feature.permute(1, 2, 0).contiguous(), logits.permute(1, 2, 0).contiguous()

class GPGait_PlusPlus(nn.Module):
    def __init__(self):
        super().__init__()
        self.build_network(dict(in_channels=[2, 64, 64, 128, 256], num_class=100, share_num=3, mask=False))


    def build_network(self, model_cfg):

        #### Configs ####

        in_c = model_cfg['in_channels']
        class_num = model_cfg['num_class']
        share_num = model_cfg['share_num']
        mask = model_cfg['mask']

        self.pose = NewNet(in_c=in_c, share_num=share_num, first_c = 2, mask=mask)
        self.bone = NewNet(in_c=in_c, share_num=share_num, first_c = 2, mask=mask)
        self.angle = NewNet(in_c=in_c, share_num=share_num, first_c = 1, joint_format = 'coco-no-head', mask=mask)

        # Head
        self.head = SeparateFCs(parts_num=19, in_channels=256, out_channels=256)
        # BNNeck
        self.BNNecks = SeparateBNNecks(class_num = class_num, in_channels=256, parts_num=19)

    def forward(self, inputs):

        ipts, labs, _, _, seqL = inputs

        x = ipts[0]
        pose = x
        N, C, T, V, M = x.size()
        if len(x.size()) == 4:
            x = x.unsqueeze(1)
        del ipts
        x_pose = x[:,:2,...]
        x_bone = x[:,2:4,...]
        x_angle = x[:,4:5,:,5:,:]
        feature_pose = self.pose(x_pose, seqL)
        feature_bone = self.bone(x_bone, seqL)
        feature_angle = self.angle(x_angle, seqL)

        feature = torch.cat([feature_pose, feature_bone, feature_angle],-1)
        # feature = feature_pose

        #Sep Fcs
        embed_1 = self.head(feature) # [n,c,p] torch.Size([256, 256, 8])

        # BNneck
        embed_2, logits = self.BNNecks(embed_1) # torch.Size([256, 256, 8])
        embed = embed_1

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed , 'labels': labs},
                'softmax':{'logits':logits,'labels':labs}
            },
            'visual_summary': {
                'image/sils': pose.view(N*T, M, V, C)
            },
            'inference_feat': {
                'embeddings': embed_2
            }
        }
        return retval

class NewNet(nn.Module):
    def __init__(self, in_c, share_num, first_c, joint_format='coco', mask=True):
        super(NewNet,self).__init__()
        in_c = in_c
        share_num = 3
        self.joint_format = joint_format

        self.graph = Graph(joint_format)

        first_c = first_c

        #### Network Define ####

        # ajaceny matrix
        self.A = torch.from_numpy(self.graph.A.astype(np.float32))
        # data normalization
        num_point = self.A.shape[-1]
        self.data_bn = nn.BatchNorm1d(first_c * num_point)

        ## layer-share ##
        layer_share = []
        layer_share.append(Part_AGCN_TCN(in_channels= first_c, out_channels= in_c[1], A = self.A, joint_format=joint_format, mask=mask))
        for i in range(1, share_num):
            layer_share.append(Part_AGCN_TCN(in_channels= in_c[i], out_channels= in_c[i+1], A = self.A, joint_format=joint_format, mask=mask))
        self.layer_share = nn.ModuleList(layer_share)

        ## global backbone ##
        backbone = []
        for i in range(share_num, len(in_c)-1):
            backbone.append(Part_AGCN_TCN(in_channels = in_c[i], out_channels = in_c[i+1], A = self.A, joint_format=joint_format, mask=mask))
        self.backbone = nn.ModuleList(backbone)

        ## part backbone ##
        part_2 = []
        for i in range(share_num, len(in_c)-1):
            part_2.append(Part_AGCN_TCN(in_channels = in_c[i], out_channels = in_c[i+1], A = self.A, joint_format=joint_format, mask=mask))
        self.part_2 = nn.ModuleList(part_2)

        if joint_format == 'coco':
            part_2_2 = []
            for i in range(share_num, len(in_c)-1):
                part_2_2.append(Part_AGCN_TCN(in_channels = in_c[i], out_channels = in_c[i+1], A = self.A, joint_format=joint_format, mask=mask))
            self.part_2_2 = nn.ModuleList(part_2_2)

            part_3_1 = []
            for i in range(share_num, len(in_c)-1):
                part_3_1.append(Part_AGCN_TCN(in_channels = in_c[i], out_channels = in_c[i+1], A = self.A, joint_format=joint_format, mask=mask))
            self.part_3_1 = nn.ModuleList(part_3_1)

        # set pooling
        self.set_pooling = PackSequenceWrapper(torch.max)

        # SPP
        self.spp = SemanticPyramidPooling(joint_format=self.joint_format)


    def forward(self, x, seqL):

        #### Network Begin ####

        # ajaceny matrix
        graph = Graph(self.joint_format)
        global_A = torch.from_numpy(graph.A.astype(np.float32))

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)

        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(
                N * M, C, T, V)

        x_share = x

        for i,m in enumerate(self.layer_share):
            x_share = m(x_share, A=global_A, part=1)

        ## part backbone ##
        x_multi_brch = []

        # 2-part-1 #
        x_this_part = x_share.clone()
        for i,m in enumerate(self.part_2):
            x_this_part = m(x_this_part, A=global_A, part=2)
        # Set Pooling
        x_this_part = self.set_pooling(x_this_part,seqL,options={"dim":2})[0]
        # V Pooling
        temp = self.spp(x_this_part, part=2)
        x_multi_brch.append(temp)

        if self.joint_format == 'coco':
            # 2-part-2 #
            x_this_part = x_share.clone()
            for i,m in enumerate(self.part_2_2):
                x_this_part = m(x_this_part, A=global_A, part=3)
            # Set Pooling
            x_this_part = self.set_pooling(x_this_part,seqL,options={"dim":2})[0]
            # V Pooling
            temp = self.spp(x_this_part, part=3)
            x_multi_brch.append(temp)

            # 3-part-1 #
            x_this_part = x_share.clone()
            for i,m in enumerate(self.part_3_1):
                x_this_part = m(x_this_part, A=global_A, part=4)
            # Set Pooling
            x_this_part = self.set_pooling(x_this_part,seqL,options={"dim":2})[0]
            # V Pooling
            temp = self.spp(x_this_part, part=4)
            x_multi_brch.append(temp)

        ## global backbone ##
        x_global = x_share.clone()
        for i,m in enumerate(self.backbone):
            x_global = m(x_global, A=global_A)
        # Set Pooling
        x_global = self.set_pooling(x_global,seqL,options={"dim":2})[0]
        # V Pooling
        temp = self.spp(x_global)
        x_multi_brch.append(temp)

        x_p = torch.cat(x_multi_brch, -1) # torch.Size([256, 256, 8])

        return x_p
