import copy as cp
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.runner import load_checkpoint
from ...utils import Graph, cache_checkpoint
from ..builder import BACKBONES
from .utils import unit_gcn, mstcn, unit_tcn

EPS = 1e-4


class GCN_Block(nn.Module):

    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, **kwargs):
        super().__init__()
        common_args = ['act', 'norm', 'g1x1']
        for arg in common_args:
            if arg in kwargs:
                value = kwargs.pop(arg)
                kwargs['tcn_' + arg] = value
                kwargs['gcn_' + arg] = value
        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {k: v for k, v in kwargs.items() if k[1:4] != 'cn_'}
        assert len(kwargs) == 0

        self.gcn = unit_gcn(in_channels, out_channels, A, **gcn_kwargs)
        self.tcn = mstcn(out_channels, out_channels, stride=stride, **tcn_kwargs)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        res = self.residual(x)
        x, gcl_graph = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), gcl_graph


"""
****************************************
*** Prototype Reconstruction Network ***
****************************************
"""  
class Prototype_Reconstruction_Network(nn.Module):
    
    def __init__(self, dim, n_prototype=100, dropout=0.1):
        super().__init__()
        self.query_matrix = nn.Linear(dim, n_prototype, bias = False)
        self.memory_matrix = nn.Linear(n_prototype, dim, bias = False)
        self.softmax = torch.softmax
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        query = self.softmax(self.query_matrix(x), dim=-1)
        z = self.memory_matrix(query)
        return self.dropout(z)


class PartAwareFusion(nn.Module):

    def __init__(self, channels, layout='nturgb+d'):
        super().__init__()
        self.layout = layout
        if layout == 'nturgb+d':
            part_inds = {
                'upper': [4, 5, 6, 7, 8, 9, 10, 11, 20, 21, 22, 23, 24],
                'lower': [0, 12, 13, 14, 15, 16, 17, 18, 19],
                'trunk': [0, 1, 2, 3, 20]
            }
        elif layout in ['openpose', 'openpose_new']:
            part_inds = {
                'upper': [1, 2, 3, 4, 5, 6, 7],
                'lower': [8, 9, 10, 11, 12, 13],
                'trunk': [0, 1, 14, 15, 16, 17]
            }
        elif layout in ['coco', 'coco_new']:
            part_inds = {
                'upper': [0, 1, 2, 3, 4, 5, 6],
                'lower': [11, 12, 13, 14, 15, 16],
                'trunk': [0, 5, 6, 11, 12]
            }
        else:
            raise ValueError(f'Unsupported layout for part fusion: {layout}')

        self.register_buffer('upper_idx', torch.tensor(part_inds['upper'], dtype=torch.long))
        self.register_buffer('lower_idx', torch.tensor(part_inds['lower'], dtype=torch.long))
        self.register_buffer('trunk_idx', torch.tensor(part_inds['trunk'], dtype=torch.long))

        self.upper_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))
        self.lower_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))
        self.trunk_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))

        self.attn = nn.Sequential(
            nn.Linear(channels * 3, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 3))

    @staticmethod
    def _pool_part(x):
        # x: [N*M, C, T, Vp] -> [N*M, C]
        return x.mean(-1).mean(-1)

    def _part_features(self, x):
        # x: [N*M, C, T, V]
        upper = self.upper_branch(x.index_select(-1, self.upper_idx))
        lower = self.lower_branch(x.index_select(-1, self.lower_idx))
        trunk = self.trunk_branch(x.index_select(-1, self.trunk_idx))
        return upper, lower, trunk

    def forward(self, x):
        # x: [N, M, C, T, V]
        N, M, C, T, V = x.shape
        x_nm = x.view(N * M, C, T, V)
        upper, lower, trunk = self._part_features(x_nm)

        upper_vec = self._pool_part(upper)
        lower_vec = self._pool_part(lower)
        trunk_vec = self._pool_part(trunk)
        gate = torch.softmax(self.attn(torch.cat([upper_vec, lower_vec, trunk_vec], dim=-1)), dim=-1)
        gate = gate.view(N * M, 3, 1, 1, 1)

        upper_full = torch.zeros_like(x_nm)
        lower_full = torch.zeros_like(x_nm)
        trunk_full = torch.zeros_like(x_nm)
        upper_full.index_copy_(-1, self.upper_idx, upper)
        lower_full.index_copy_(-1, self.lower_idx, lower)
        trunk_full.index_copy_(-1, self.trunk_idx, trunk)

        fused = gate[:, 0] * upper_full + gate[:, 1] * lower_full + gate[:, 2] * trunk_full
        fused = fused.view(N, M, C, T, V)
        return x + fused


@BACKBONES.register_module()
class ProtoGCN(nn.Module):

    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=96,
                 ch_ratio=2,
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 data_bn_type='VC',
                 num_person=2,
                 pretrained=None,
                 **kwargs):
        super().__init__()

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn_type = data_bn_type
        self.layout = graph_cfg.get('layout', 'nturgb+d')
        self.use_part_fusion = kwargs.pop('use_part_fusion', False)
        self.kwargs = kwargs

        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        num_prototype = kwargs.pop('num_prototype', 100)
        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)
        lw_kwargs[0].pop('g1x1', None)
        lw_kwargs[0].pop('gcn_g1x1', None)

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages
        modules = []
        if self.in_channels != self.base_channels:
            modules = [GCN_Block(in_channels, base_channels, A.clone(), 1, residual=False, **lw_kwargs[0])]

        inflate_times = 0
        down_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(GCN_Block(in_channels, out_channels, A.clone(), stride, **lw_kwargs[i - 1]))
            down_times += (i in down_stages)

        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.pretrained = pretrained
        
        out_channels = base_channels
        norm = 'BN'
        norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        
        self.post = nn.Conv2d(out_channels, out_channels, 1)
        self.bn = build_norm_layer(norm_cfg, out_channels)[1]
        self.relu = nn.ReLU()
        if self.use_part_fusion:
            self.part_fusion = PartAwareFusion(out_channels, layout=self.layout)
        
        dim = 384   # base_channels * 4
        self.prn = Prototype_Reconstruction_Network(dim, num_prototype)
        
    def init_weights(self):
        if isinstance(self.pretrained, str):
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=False)

    def forward(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        get_graph = []
        for i in range(self.num_stages):
            x, gcl_graph = self.gcn[i](x)
            # N*M C V V
            get_graph.append(gcl_graph)
        
        x = x.reshape((N, M) + x.shape[1:])
        if self.use_part_fusion:
            x = self.part_fusion(x)
        c_graph = x.size(2)
        
        graph = get_graph[-1]
        # N C V V -> N C V*V
        graph = graph.view(N, M, c_graph, V, V).mean(1).view(N, c_graph, V * V)
        
        the_graph_list = []
        for i in range(N):
            # V*V C
            the_graph = graph[i].permute(1, 0)
            # V*V C
            the_graph = self.prn(the_graph)
            # C V V
            the_graph = the_graph.permute(1, 0).view(c_graph, V, V)
            the_graph_list.append(the_graph)
        
        # N C V V
        re_graph = torch.stack(the_graph_list, dim=0)
        re_graph = self.post(re_graph)
        reconstructed_graph = self.relu(self.bn(re_graph))
        # N V*V
        reconstructed_graph = reconstructed_graph.mean(1).view(N, -1)
        
        return x, reconstructed_graph
