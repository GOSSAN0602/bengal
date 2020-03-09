import sys
import numpy as np
sys.path.append('./')
from libs.lazy_utils import *
from libs.mixup_cutmix import *
from torch import nn
import torch.nn.functional as F
import pretrainedmodels
from torch.nn import Sequential


class LinearBlock(nn.Module):

    def __init__(self, in_features, out_features, bias=True,
                 use_bn=True, activation=F.relu, dropout_ratio=-1, residual=False,):
        super(LinearBlock, self).__init__()
        if in_features is None:
            self.linear = LazyLinear(in_features, out_features, bias=bias)
        else:
            self.linear = nn.Linear(in_features, out_features, bias=bias)
        if use_bn:
            self.bn = nn.BatchNorm1d(out_features)
        if dropout_ratio > 0.:
            self.dropout = nn.Dropout(p=dropout_ratio)
        else:
            self.dropout = None
        self.activation = activation
        self.use_bn = use_bn
        self.dropout_ratio = dropout_ratio
        self.residual = residual

    def __call__(self, x):
        h = self.linear(x)
        if self.use_bn:
            h = self.bn(h)
        if self.activation is not None:
            h = self.activation(h)
        if self.residual:
            h = residual_add(h, x)
        if self.dropout_ratio > 0:
            h = self.dropout(h)
        return h

class PretrainedCNN(nn.Module):
    def __init__(self, model_name='se_resnext101_32x4d',
                 in_channels=1, out_dim=10, use_bn=True,
                 pretrained='imagenet'):
        super(PretrainedCNN, self).__init__()
        self.conv0 = nn.Conv2d(
            in_channels, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.base_model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
        activation = F.leaky_relu
        self.do_pooling = True
        if self.do_pooling:
            inch = self.base_model.last_linear.in_features
        else:
            inch = None
        hdim = 512
        lin1 = LinearBlock(inch, hdim, use_bn=use_bn, activation=activation, residual=False)
        lin2 = LinearBlock(hdim, out_dim, use_bn=use_bn, activation=None, residual=False)
        self.lin_layers = Sequential(lin1, lin2)

    def forward(self, x):
        h = self.conv0(x)
        h = self.base_model.features(h)

        if self.do_pooling:
            h = torch.sum(h, dim=(-1, -2))
        else:
            # [128, 2048, 4, 4] when input is (128, 128)
            bs, ch, height, width = h.shape
            h = h.view(bs, ch*height*width)
        for layer in self.lin_layers:
            h = layer(h)
        return h

def accuracy(y, t):
    pred_label = torch.argmax(y, dim=1)
    count = pred_label.shape[0]
    correct = (pred_label == t).sum().type(torch.float32)
    acc = correct / count
    return acc


class BengaliClassifier(nn.Module):
    def __init__(self, predictor, n_grapheme=168, n_vowel=11, n_consonant=7):
        super(BengaliClassifier, self).__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        self.n_total_class = self.n_grapheme + self.n_vowel + self.n_consonant
        self.predictor = predictor

        self.metrics_keys = [
            'ohem_loss', 'ohem_loss_grapheme', 'ohem_loss_vowel', 'ohem_loss_consonant', 'cr_loss',
            'cr_loss_grapheme', 'cr_loss_vowel', 'cr_loss_consonant', 'acc_grapheme', 'acc_vowel', 'acc_consonant']

    def forward(self, x, y=None):
        images = x
        label1 = y[:,0]
        label2 = y[:,1]
        label3 = y[:,2]
        if np.random.rand() < 0.5:
            images, targets = mixup(images, label1, label2, label3, 0.4)
            pred = self.predictor(x)
            if isinstance(pred, tuple):
                assert len(pred) == 3
                preds = pred
            else:
                assert pred.shape[1] == self.n_total_class
                preds = torch.split(pred, [self.n_grapheme, self.n_vowel, self.n_consonant], dim=1)
            output1, output2, output3 = preds[0], preds[1], preds[2]
            loss = mixup_criterion(output1, output2, output3, targets)
        else:
            images, targets = cutmix(images, label1, label2, label3, 0.4)
            pred = self.predictor(x)
            if isinstance(pred, tuple):
                assert len(pred) == 3
                preds = pred
            else:
                assert pred.shape[1] == self.n_total_class
                preds = torch.split(pred, [self.n_grapheme, self.n_vowel, self.n_consonant], dim=1)
            output1, output2, output3 = preds[0], preds[1], preds[2]
            loss = cutmix_criterion(output1, output2, output3, targets)

        loss_grapheme = F.cross_entropy(preds[0], y[:, 0])
        loss_vowel = F.cross_entropy(preds[1], y[:, 1])
        loss_consonant = F.cross_entropy(preds[2], y[:, 2])
        loss_sum = loss_grapheme + loss_vowel + loss_consonant
        metrics = {
            'ohem_loss': (loss[0]+loss[1]+loss[2]).item(),
            'ohem_loss_grapheme': loss[0].item(),
            'ohem_loss_vowel': loss[1].item(),
            'ohem_loss_consonant': loss[2].item(),
            'cr_loss': loss_sum.item(),
            'cr_loss_grapheme': loss_grapheme.item(),
            'cr_loss_vowel': loss_vowel.item(),
            'cr_loss_consonant': loss_consonant.item(),
            'acc_grapheme': accuracy(preds[0], y[:, 0]),
            'acc_vowel': accuracy(preds[1], y[:, 1]),
            'acc_consonant': accuracy(preds[2], y[:, 2]),
        }
        return loss[0]+loss[1]+loss[2], metrics, pred

    def calc(self, data_loader):
        device: torch.device = next(self.parameters()).device
        self.eval()
        output_list = []
        with torch.no_grad():
            for batch in tqdm(data_loader):
                # Caution: support general preprocessing.
                # If `data` is not `Data` instance, `to` method is not supported!
                batch = batch.to(device)
                pred = self.predictor(batch)
                output_list.append(pred)
        output = torch.cat(output_list, dim=0)
        preds = torch.split(output, [self.n_grapheme, self.n_vowel, self.n_consonant], dim=1)
        return preds

    def predict_proba(self, data_loader):
        preds = self.calc(data_loader)
        return [F.softmax(p, dim=1) for p in preds]

    def predict(self, data_loader):
        preds = self.calc(data_loader)
        pred_labels = [torch.argmax(p, dim=1) for p in preds]
        return pred_labels
