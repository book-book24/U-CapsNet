from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import numpy as np
from PIL import Image
from scipy.io import savemat

parser = argparse.ArgumentParser(description='U_CapsNet Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=3000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
# parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                     help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args.cuda)
torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)  # 矩阵的转置
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)),
                                 dim=-1)  # contiguous()将内存变为连续的 size(-1)右边的维度
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


def augmentation(x, max_shift=2):
    _, _, height, width = x.size()

    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)

    shifted_image = torch.zeros(*x.size())
    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
    return shifted_image.float()


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=3):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels,
                                                          out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:

            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
            logits = Variable(torch.zeros(*priors.size()))
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)
        return outputs


class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=10, num_route_nodes=-1, in_channels=256, out_channels=64,
                                             kernel_size=9, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=1, num_route_nodes=64 * 7 * 7, in_channels=10,
                                           out_channels=40)
        self.decoder_input = nn.Linear(40, 1024)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 64, 9, 1, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 256, 9, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 3, 4, 1),
            nn.Tanh())

    def forward(self, x, train=True):
        x1 = F.relu(self.conv1(x), inplace=True)
        x2 = self.primary_capsules(x1)
        x3 = self.digit_capsules(x2).transpose(0, 1)
        x4 = self.decoder_input(x3)
        x5 = x4.view(-1, 16, 8, 8)
        reconstructions = self.decoder(x5)
        # reconstructions = reconstructions.view(-1, 30 * 30 * 3)
        if not train:
            savemat('./data/CNP/u_capsnet_feat.mat', {'L1': x1.data.cpu().numpy(), 'L2': x2.data.cpu().numpy(),
                                                      'L3': x3.data.cpu().numpy()})

        return reconstructions


process = transforms.ToTensor()


class img_dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.images = [os.path.join(path, img) for img in os.listdir(path)]

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        img = img.resize((30, 30))

        return process(img)

    def __len__(self):
        return len(self.images)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        # target = torch.eye(4).index_select(dim=0, index=target)
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, data)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data, False)
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n], output[:n]])
            save_image(comparison.cpu(), './data/CNP/reconstruction' + '.png', nrow=n)


train_data = img_dataset('./data/CNP/train')
test_data = img_dataset('./data/CNP/test')
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
Net = CapsuleNet().to(device)
optimizer = optim.Adam(Net.parameters(), lr=1e-3)

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(args, Net, device, train_loader, optimizer, epoch)

    torch.save(Net.state_dict(), './data/CNP/epoch_%d.pt' % epoch)
    test(Net, device, test_loader)
