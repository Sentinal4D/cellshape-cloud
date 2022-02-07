import torch

from dataloaders.pointcloud_dataset import *
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime
from naming_and_reports import *
from chamfer import *


def create_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use DCEC for clustering')
    parser.add_argument('--dataset_path', default='./', type=str)
    parser.add_argument('--dataframe_path', default='./', type=str)
    parser.add_argument('--output_dir', default='./', type=str)
    parser.add_argument('--num_features', default=50, type=int)
    parser.add_argument('--shape', default='plane', type=str)
    parser.add_argument('--load_path', default=None)
    parser.add_argument('--learning_rate', default=0.00001, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--decoder', default='tearing', type=str)
    parser.add_argument('--train_on_all', default=False, type=bool)
    parser.add_argument('--component', default='cell', type=str)

    args = parser.parse_args()
    df = args.dataframe_path
    root_dir = args.dataset_path
    output_dir = args.output_dir
    num_features = args.num_features
    shape = args.shape
    load_path = args.load_path
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    decoder = args.decoder
    train_all = args.train_on_all
    cell_component = args.component

    model_name = f'FoldingNetNew_{num_features}feats_{shape}shape_{decoder}decoder_trainall{train_all}'
    f, name_net, saved_to, name_txt, name = reports(model_name, output_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    to_eval = "ReconstructionNet" + "(" + f"'{decoder}'" + ", num_clusters=5, " \
                                                                        "num_features=num_features, " \
                                                                        "shape=shape)"
    model = eval(to_eval)
    model = model.to(device)

    print_both(f, f'Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate * 16 / batch_size,
                                 betas=(0.9, 0.999),
                                 weight_decay=1e-6)

    if load_path:
        try:
            model.load_state_dict(torch.load(load_path)['model_state_dict'])
            optimizer.load_state_dict(torch.load(load_path)['optimizer_state_dict'])
            print_both(f, 'Loading model from ' + load_path)
        except:
            print_both(f, 'Model either does not exist or is the wrong path.')

    # Data loaders
    if train_all:
        dataset = PointCloudDatasetAll(df,
                                       root_dir,
                                       transform=None,
                                       img_size=400,
                                       target_transform=True,
                                       cell_component=cell_component)
    else:
        dataset = PointCloudDataset(df,
                                    root_dir,
                                    transform=None,
                                    img_size=400,
                                    target_transform=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader_inf = DataLoader(dataset, batch_size=1, shuffle=False)

    # Optimisers and schedulers
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0000001, momentum=0.9)
    criterion = ChamferLoss1()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.00000001, weight_decay=0)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 0.0000001, 0.001)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)

    # Logging
    writer_logging = output_dir + 'runs/' + name
    writer = SummaryWriter(log_dir=writer_logging)
    date_time = str(datetime.datetime.now()).replace(" ", "_").replace("-", "_")[:-7]
    num_epochs = 500

    print_both(f, 'Date: ' + date_time)
    print_both(f, 'Training model: ' + name)
    print_both(f, 'Number of epochs: {}'.format(num_epochs))
    print_both(f, 'Number of features: {}'.format(num_features))
    print_both(f, 'Shape used: {}'.format(shape))
    print_both(f, 'Output directory of nets: ' + name_net + '.pt')
    print_both(f, 'Output directory of reports ' + name_txt)
    print_both(f, 'Output directory of tensorboard logging ' + writer_logging)
    f.close()

    total_loss = 0.
    rec_loss = 0.
    clus_loss = 0.
    model.train()
    threshold = 0.
    losses = []
    test_acc = []
    best_acc = 0.
    best_loss = 1000000000
    niter = 1
    for epoch in range(num_epochs):
        batch_num = 1
        running_loss = 0.
        f = open(name_txt, 'a')
        print_both(f, 'Training epoch {}'.format(epoch))
        f.close()
        model.train()
        batches = []

        for i, data in enumerate(dataloader, 0):
            inputs, labels, _ = data
            inputs = inputs.to(device)

            # ===================forward=====================
            with torch.set_grad_enabled(True):
                output, feature, embedding, clustering_out, fold1 = model(inputs)
                optimizer.zero_grad()
                loss = model.get_loss(inputs, output)
                # ===================backward====================
                loss.backward()
                optimizer.step()

            running_loss += loss.detach().item()/batch_size
            batch_num += 1
            writer.add_scalar('/Loss' + 'Batch', loss.detach().item()/batch_size, niter)
            niter += 1

            lr = np.asarray(optimizer.param_groups[0]['lr'])

            if i % 10 == 0:
                f = open(name_txt, 'a')
                print_both(f, '[%d/%d][%d/%d]\tLossTot: %.4f\tLossRec: %.4f' % (epoch,
                                                                                num_epochs,
                                                                                i,
                                                                                len(dataloader),
                                                                                loss.detach().item()/batch_size,
                                                                                loss.detach().item()/batch_size,))

                f.close()
                # points = output[0].cpu().detach().numpy()
                # image = plot_to_image(plot_point_cloud(points))
                # writer.add_image("/Output point cloud{}".format(niter), image, niter)

        # ===================log========================
        total_loss = running_loss/len(dataloader)
        if total_loss < best_loss:
            checkpoint = {'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'epoch': epoch,
                          'loss': total_loss}
            best_loss = total_loss
            create_dir_if_not_exist(output_dir)
            f = open(name_txt, 'a')
            print_both(f, 'Saving model to:' + name_net + '.pt' + ' with loss = {}'
                  .format(total_loss) + ' at epoch {}'.format(epoch))
            torch.save(checkpoint, name_net + '.pt')
            print_both(f, 'epoch [{}/{}], loss:{}'.format(epoch + 1, num_epochs, total_loss))
            f.close()

        f = open(name_txt, 'a')
        print_both(f, 'epoch [{}/{}], loss:{:.4f}, Rec loss:{:.4f}'.format(epoch + 1,
                                                                           num_epochs,
                                                                           total_loss,
                                                                           total_loss))
        f.close()

