import torch.nn as nn
import torch
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_and_init_FC_layer(din, dout):
    li = nn.Linear(din, dout)
    # init weights/bias
    nn.init.xavier_uniform_(li.weight.data, gain=nn.init.calculate_gain('relu'))
    li.bias.data.fill_(0.)
    return li


def get_MLP_layers(dims, doLastRelu):
    layers = []
    for i in range(1, len(dims)):
        layers.append(get_and_init_FC_layer(dims[i - 1], dims[i]))
        if i == len(dims) - 1 and not doLastRelu:
            continue
        layers.append(nn.ReLU())
    return layers


class PointwiseMLP(nn.Sequential):
    '''Nxdin ->Nxd1->Nxd2->...-> Nxdout'''

    def __init__(self, dims, doLastRelu=False):
        layers = get_MLP_layers(dims, doLastRelu)
        super(PointwiseMLP, self).__init__(*layers)


class GlobalPool(nn.Module):
    '''BxNxK -> BxK'''

    def __init__(self, pool_layer):
        super(GlobalPool, self).__init__()
        self.Pool = pool_layer

    def forward(self, X):
        X = X.unsqueeze(-3)
        X = self.Pool(X)
        X = X.squeeze(-2)
        X = X.squeeze(-2)
        return X


class PointNetGlobalMax(nn.Sequential):
    '''BxNxdims[0] -> Bxdims[-1]'''

    def __init__(self, dims, doLastRelu=False):
        layers = [
            PointwiseMLP(dims, doLastRelu=doLastRelu),
            GlobalPool(nn.AdaptiveMaxPool2d((1, dims[-1]))),
        ]
        super(PointNetGlobalMax, self).__init__(*layers)


class PointNetVanilla(nn.Sequential):

    def __init__(self, MLP_dims, FC_dims, MLP_doLastRelu):
        assert (MLP_dims[-1] == FC_dims[0])
        layers = [
            PointNetGlobalMax(MLP_dims, doLastRelu=MLP_doLastRelu),
        ]
        layers.extend(get_MLP_layers(FC_dims, False))
        super(PointNetVanilla, self).__init__(*layers)


class PointNetVanillaModel(nn.Module):

    @staticmethod
    def add_options(parser, isTrain=True):
        parser.add_argument('--pointnet_mlp_dims', type=int, nargs='+', default=[3, 64, 128, 128, 1024],
                            help='Dimensions of the MLP in the PointNet encoder.')
        parser.add_argument('--pointnet_fc_dims', type=int, nargs='+', default=[1024, 512, 512, 512],
                            help='Dimensions of the FC in the PointNet encoder.')
        parser.add_argument("--pointnet_mlp_dolastrelu", type=str2bool, nargs='?', const=True, default=False,
                            help='Apply the last ReLU or not in the PointNet encoder.')
        return parser

    def __init__(self, opt):
        super(PointNetVanillaModel, self).__init__()
        self.pointnet = PointNetVanilla(opt.pointnet_mlp_dims, opt.pointnet_fc_dims, opt.pointnet_mlp_dolastrelu)

    def forward(self, data):
        return self.pointnet(data)


class FoldingNetVanilla(nn.Module):

    def __init__(self, folding1_dims, folding2_dims):
        super(FoldingNetVanilla, self).__init__()

        # The folding decoder
        self.fold1 = PointwiseMLP(folding1_dims, doLastRelu=False)
        if folding2_dims[0] > 0:
            self.fold2 = PointwiseMLP(folding2_dims, doLastRelu=False)
        else:
            self.fold2 = None

    def forward(self, cw, grid, **kwargs):

        cw_exp = cw.unsqueeze(1).expand(-1, grid.shape[1], -1)  # batch_size X point_num X code_length

        # 1st folding
        in1 = torch.cat((grid, cw_exp), 2)  # batch_size X point_num X (code_length + 3)
        out1 = self.fold1(in1)  # batch_size X point_num X 3

        # 2nd folding
        if not (self.fold2 is None):
            in2 = torch.cat((out1, cw_exp), 2)  # batch_size X point_num X (code_length + 4)
            out2 = self.fold2(in2)  # batch_size X point_num X 3
            return out2
        else:
            return out1


class FoldingNetVanillaModel(nn.Module):

    @staticmethod
    def add_options(parser, isTrain=True):
        # Some optionals
        parser.add_argument('--grid_dims', type=int, nargs='+', default=[45, 45], help='Grid dimensions.')
        parser.add_argument('--folding1_dims', type=int, nargs='+', default=[514, 512, 512, 3],
                            help='Dimensions of the first folding module.')
        parser.add_argument('--folding2_dims', type=int, nargs='+', default=[515, 512, 512, 3],
                            help='Dimensions of the second folding module.')
        return parser

    def __init__(self, opt):
        super(FoldingNetVanillaModel, self).__init__()

        # Initialize the 2D grid
        range_x = torch.linspace(-1.0, 1.0, opt.grid_dims[0])
        range_y = torch.linspace(-1.0, 1.0, opt.grid_dims[1])
        x_coor, y_coor = torch.meshgrid(range_x, range_y)
        self.grid = torch.stack([x_coor, y_coor], axis=-1).float().reshape(-1, 2)

        # Initialize the folding module
        self.folding1 = FoldingNetVanilla(opt.folding1_dims, opt.folding2_dims)

    def forward(self, cw):
        grid = self.grid.cuda().unsqueeze(0).expand(cw.shape[0], -1, -1)  # batch_size X point_num X 2
        pc = self.folding1(cw, grid)
        return pc


class GraphFilter(nn.Module):

    def __init__(self, grid_dims=[45, 45], graph_r=1e-12, graph_eps=0.02, graph_lam=0.5):
        super(GraphFilter, self).__init__()
        self.grid_dims = grid_dims
        self.graph_r = graph_r
        self.graph_eps_sqr = graph_eps * graph_eps
        self.graph_lam = graph_lam

    def forward(self, grid, pc):
        # Data preparation
        bs_cur = pc.shape[0]
        grid_exp = grid.contiguous().view(bs_cur, self.grid_dims[0], self.grid_dims[1],
                                          2)  # batch_size X dim0 X dim1 X 2
        pc_exp = pc.contiguous().view(bs_cur, self.grid_dims[0], self.grid_dims[1], 3)  # batch_size X dim0 X dim1 X 3
        graph_feature = torch.cat((grid_exp, pc_exp), dim=3).permute([0, 3, 1, 2])

        # Compute the graph weights
        wght_hori = graph_feature[:, :, :-1, :] - graph_feature[:, :, 1:, :]  # horizontal weights
        wght_vert = graph_feature[:, :, :, :-1] - graph_feature[:, :, :, 1:]  # vertical weights
        wght_hori = torch.exp(-torch.sum(wght_hori * wght_hori, dim=1) / self.graph_eps_sqr)  # Gaussian weight
        wght_vert = torch.exp(-torch.sum(wght_vert * wght_vert, dim=1) / self.graph_eps_sqr)
        wght_hori = (wght_hori > self.graph_r) * wght_hori
        wght_vert = (wght_vert > self.graph_r) * wght_vert
        wght_lft = torch.cat((torch.zeros([bs_cur, 1, self.grid_dims[1]]).cuda(), wght_hori), 1)  # add left
        wght_rgh = torch.cat((wght_hori, torch.zeros([bs_cur, 1, self.grid_dims[1]]).cuda()), 1)  # add right
        wght_top = torch.cat((torch.zeros([bs_cur, self.grid_dims[0], 1]).cuda(), wght_vert), 2)  # add top
        wght_bot = torch.cat((wght_vert, torch.zeros([bs_cur, self.grid_dims[0], 1]).cuda()), 2)  # add bottom
        wght_all = torch.cat(
            (wght_lft.unsqueeze(1), wght_rgh.unsqueeze(1), wght_top.unsqueeze(1), wght_bot.unsqueeze(1)), 1)

        # Perform the actural graph filtering: x = (I - \lambda L) * x
        wght_hori = wght_hori.unsqueeze(1).expand(-1, 3, -1, -1)  # dimension expansion
        wght_vert = wght_vert.unsqueeze(1).expand(-1, 3, -1, -1)
        pc = pc.permute([0, 2, 1]).contiguous().view(bs_cur, 3, self.grid_dims[0], self.grid_dims[1])
        pc_filt = \
            torch.cat((torch.zeros([bs_cur, 3, 1, self.grid_dims[1]]).cuda(), pc[:, :, :-1, :] * wght_hori), 2) + \
            torch.cat((pc[:, :, 1:, :] * wght_hori, torch.zeros([bs_cur, 3, 1, self.grid_dims[1]]).cuda()), 2) + \
            torch.cat((torch.zeros([bs_cur, 3, self.grid_dims[0], 1]).cuda(), pc[:, :, :, :-1] * wght_vert), 3) + \
            torch.cat((pc[:, :, :, 1:] * wght_vert, torch.zeros([bs_cur, 3, self.grid_dims[0], 1]).cuda()),
                      3)  # left, right, top, bottom

        pc_filt = pc + self.graph_lam * (pc_filt - torch.sum(wght_all, dim=1).unsqueeze(1).expand(-1, 3, -1,
                                                                                                  -1) * pc)  # equivalent to ( I - \lambda L) * x
        pc_filt = pc_filt.view(bs_cur, 3, -1).permute([0, 2, 1])
        return pc_filt, wght_all


def get_Conv2d_layer(dims, kernel_size, doLastRelu):
    layers = []
    for i in range(1, len(dims)):
        if kernel_size != 1:
            layers.append(nn.ReplicationPad2d(int((kernel_size - 1) / 2)))
        layers.append(nn.Conv2d(in_channels=dims[i - 1], out_channels=dims[i],
                                kernel_size=kernel_size, stride=1, padding=0, bias=True))
        if i == len(dims) - 1 and not doLastRelu:
            continue
        layers.append(nn.ReLU(inplace=True))
    return layers


class Conv2dLayers(nn.Sequential):
    def __init__(self, dims, kernel_size, doLastRelu=False):
        layers = get_Conv2d_layer(dims, kernel_size, doLastRelu)
        super(Conv2dLayers, self).__init__(*layers)


class TearingNetBasic(nn.Module):

    def __init__(self, tearing1_dims, tearing2_dims, grid_dims=[45, 45], kernel_size=1):
        super(TearingNetBasic, self).__init__()

        self.grid_dims = grid_dims
        self.tearing1 = Conv2dLayers(tearing1_dims, kernel_size=kernel_size, doLastRelu=False)
        self.tearing2 = Conv2dLayers(tearing2_dims, kernel_size=kernel_size, doLastRelu=False)

    def forward(self, cw, grid, pc, **kwargs):
        grid_exp = grid.contiguous().view(grid.shape[0], self.grid_dims[0], self.grid_dims[1],
                                          2)  # batch_size X dim0 X dim1 X 2
        pc_exp = pc.contiguous().view(pc.shape[0], self.grid_dims[0], self.grid_dims[1],
                                      3)  # batch_size X dim0 X dim1 X 3
        cw_exp = cw.unsqueeze(1).unsqueeze(1).expand(-1, self.grid_dims[0], self.grid_dims[1],
                                                     -1)  # batch_size X dim0 X dim1 X code_length
        in1 = torch.cat((grid_exp, pc_exp, cw_exp), 3).permute([0, 3, 1, 2])

        # Compute the torn 2D grid
        out1 = self.tearing1(in1)  # 1st tearing
        in2 = torch.cat((in1, out1), 1)
        out2 = self.tearing2(in2)  # 2nd tearing
        out2 = out2.permute([0, 2, 3, 1]).contiguous().view(grid.shape[0], self.grid_dims[0] * self.grid_dims[1], 2)
        return grid + out2


class TearingNetBasicModel(nn.Module):

    @staticmethod
    def add_options(parser, isTrain=True):
        # General optional(s)
        parser.add_argument('--grid_dims', type=int, nargs='+', default=[45, 45], help='Grid dimensions.')

        # Options related to the Folding Network
        parser.add_argument('--folding1_dims', type=int, nargs='+', default=[514, 512, 512, 3],
                            help='Dimensions of the first folding module.')
        parser.add_argument('--folding2_dims', type=int, nargs='+', default=[515, 512, 512, 3],
                            help='Dimensions of the second folding module.')

        # Options related to the Tearing Network
        parser.add_argument('--tearing1_dims', type=int, nargs='+', default=[523, 256, 128, 64],
                            help='Dimensions of the first tearing module.')
        parser.add_argument('--tearing2_dims', type=int, nargs='+', default=[587, 256, 128, 2],
                            help='Dimensions of the second tearing module.')
        parser.add_argument('--tearing_conv_kernel_size', type=int, default=1,
                            help='Kernel size of the convolutional layers in the Tearing Network, 1 implies MLP.')

        return parser

    def __init__(self, opt):
        super(TearingNetBasicModel, self).__init__()

        # Initialize the regular 2D grid
        range_x = torch.linspace(-1.0, 1.0, 45)
        range_y = torch.linspace(-1.0, 1.0, 45)
        x_coor, y_coor = torch.meshgrid(range_x, range_y)
        self.grid = torch.stack([x_coor, y_coor], axis=-1).float().reshape(-1, 2)

        # Initialize the Folding Network and the Tearing Network
        self.folding = FoldingNetVanilla([514, 512, 512, 3], [515, 512, 512, 3])
        self.tearing = TearingNetBasic(opt.tearing1_dims, opt.tearing2_dims, opt.grid_dims,
                                       opt.tearing_conv_kernel_size)
        self.deembedding = nn.Linear(50, 512, bias=False)

    def forward(self, cw):
        cw = self.deembedding(cw)
        cw = cw.unsqueeze(1)
        grid0 = self.grid.cuda().unsqueeze(0).expand(cw.shape[0], -1, -1)  # batch_size X point_num X 2
        pc0 = self.folding(cw, grid0)  # Folding Network
        grid1 = self.tearing(cw, grid0, pc0)  # Tearing Network
        pc1 = self.folding(cw, grid1)  # Folding Network

        return pc0, pc1, grid1


class TearingNetGraphModel(nn.Module):

    @staticmethod
    def add_options(parser, isTrain=True):
        # General optional(s)
        parser.add_argument('--grid_dims', type=int, nargs='+', default=[45, 45], help='Grid dimensions.')

        # Options related to the Folding Network
        parser.add_argument('--folding1_dims', type=int, nargs='+', default=[514, 512, 512, 3],
                            help='Dimensions of the first folding module.')
        parser.add_argument('--folding2_dims', type=int, nargs='+', default=[515, 512, 512, 3],
                            help='Dimensions of the second folding module.')

        # Options related to the Tearing Network
        parser.add_argument('--tearing1_dims', type=int, nargs='+', default=[523, 256, 128, 64],
                            help='Dimensions of the first tearing module.')
        parser.add_argument('--tearing2_dims', type=int, nargs='+', default=[587, 256, 128, 2],
                            help='Dimensions of the second tearing module.')
        parser.add_argument('--tearing_conv_kernel_size', type=int, default=1,
                            help='Kernel size of the convolutional layers in the Tearing Network, 1 implies MLP.')

        # Options related to graph construction
        parser.add_argument('--graph_r', type=float, default=1e-12, help='Parameter r for the r-neighborhood graph.')
        parser.add_argument('--graph_eps', type=float, default=0.02,
                            help='Parameter epsilon for the graph (bandwidth parameter).')
        parser.add_argument('--graph_lam', type=float, default=0.5, help='Parameter lambda for the graph filter.')

        return parser

    def __init__(self):
        super(TearingNetGraphModel, self).__init__()

        # Initialize the regular 2D grid
        range_x = torch.linspace(-1.0, 1.0, 45)
        range_y = torch.linspace(-1.0, 1.0, 45)
        x_coor, y_coor = torch.meshgrid(range_x, range_y)
        self.grid = torch.stack([x_coor, y_coor], axis=-1).float().reshape(-1, 2)

        # Initialize the Folding Network and the Tearing Network
        self.folding = FoldingNetVanilla([514, 512, 512, 3], [515, 512, 512, 3])
        self.tearing = TearingNetBasic([517, 512, 512, 64], [581, 512, 512, 2], [45, 45], 1)
        self.graph_filter = GraphFilter([45, 45], 1e-12, 0.02, 0.5)
        self.deembedding = nn.Linear(50, 512, bias=False)

    def forward(self, cw):
        cw = self.deembedding(cw)
        #         cw = cw.unsqueeze(1)
        grid0 = self.grid.cuda().unsqueeze(0).expand(cw.shape[0], -1, -1)  # batch_size X point_num X 2

        pc0 = self.folding(cw, grid0)  # Folding Network
        grid1 = self.tearing(cw, grid0, pc0)  # Tearing Network
        pc1 = self.folding(cw, grid1)  # Folding Network
        pc2, graph_wght = self.graph_filter(grid1, pc1)  # Graph Filtering
        return pc0, pc1, pc2, grid1, graph_wght