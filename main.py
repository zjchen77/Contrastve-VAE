import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
from torchvision.utils import save_image

import utils1
from model1 import Model



# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, target in train_bar:
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        out_1, recon_x1, mu1, logvar1 = net(pos_1)
        out_2, recon_x2, mu2, logvar2 = net(pos_2)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        
        MSE = nn.MSELoss(size_average=False)
        
        BCE = MSE(recon_x1, pos_1) # mse loss
        # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mu1.pow(2).add_(logvar1.exp()).mul_(-1).add_(1).add_(logvar1)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        # KL divergence
        
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        

        L1 = total_loss / total_num   # NT-Xent
        L2 =  BCE + KLD
        final_loss  = Alpha*L1 + Beta*L2
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, final_loss))
        
        return final_loss


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):     #测试集，算 Top1 和 Top5
    net.eval()
    
    
    with torch.no_grad():
            
        noise = Variable(torch.randn(1,512,32,32))  #noise 尺寸瞎写的
        d = Model(512).decoder
        f = Model(32).final_layer
        out = d(noise)
        out = f(out)
        save_image(out, './SimVae_img/image.png')

    return noise, out 


if __name__ == '__main__':    
#这里调用了 train 和 test
#当.py文件被直接运行时，if __name__ == '__main__'之下的代码块将被运行；
#当.py文件以模块形式被导入时，if __name__ == '__main__'之下的代码块不被运行。

    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=512, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--Alpha', default=500, type=float, help='Coefficient of NT-Xent')
    parser.add_argument('--Beta', default=500, type=float, help='Coefficient of reconstruction loss')
    #添加了alpha和beta做超参数
    
    # args parse
    args = parser.parse_args()
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs
    Alpha, Beta = args.Alpha, args.Beta   

    # data prepare
    train_data = utils1.CIFAR10Pair(root='data', train=True, transform=utils1.train_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                              drop_last=True)
    memory_data = utils1.CIFAR10Pair(root='data', train=True, transform=utils1.test_transform, download=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_data = utils1.CIFAR10Pair(root='data', train=False, transform=utils1.test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # model setup and optimizer config
    model = Model(feature_dim).cuda()
    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    c = len(memory_data.classes)

    # training loop
    results = {'train_loss': [], 'noise': [], 'out': []}
    save_name_pre = '{}_{}_{}_{}_{}'.format(feature_dim, temperature, k, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        noise, out = test(model, memory_loader, test_loader)
        results['noise'].append(noise)
        results['out'].append(out)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        
