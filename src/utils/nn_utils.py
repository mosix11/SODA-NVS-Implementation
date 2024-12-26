import torch
import os
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import collections
from . import misc_utils
from torch.utils.data import Dataset
from typing import Any
from sklearn.metrics import confusion_matrix

def get_gpu_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else : return None
    
def get_cpu_device():
    return torch.device('cpu')


class ClassifierMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0
        self.true_positives = torch.zeros(self.num_classes)
        self.predicted_positives = torch.zeros(self.num_classes)
        self.actual_positives = torch.zeros(self.num_classes)

    def update(self, preds, labels):
        _, preds_max = torch.max(preds, 1)
        self.correct += (preds_max == labels).sum().item()
        self.total += labels.size(0)

        for i in range(self.num_classes):
            self.true_positives[i] += ((preds_max == i) & (labels == i)).sum().item()
            self.predicted_positives[i] += (preds_max == i).sum().item()
            self.actual_positives[i] += (labels == i).sum().item()

    def accuracy(self):
        return self.correct / self.total

    def precision(self):
        precision_per_class = self.true_positives / (self.predicted_positives + 1e-10)
        return precision_per_class.mean().item()

    def recall(self):
        recall_per_class = self.true_positives / (self.actual_positives + 1e-10)
        return recall_per_class.mean().item()

    def f1_score(self):
        prec = self.precision()
        rec = self.recall()
        return 2 * (prec * rec) / (prec + rec + 1e-10)

class ConfusionMatrixTracker:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int32)

    def update(self, predicted_labels, true_labels):
        # Ensure labels are detached, moved to CPU, and converted to numpy arrays
        true_labels_np = true_labels.detach().cpu().numpy()
        predicted_labels_np = predicted_labels.detach().cpu().numpy()
        if len(true_labels_np.shape) > 1 : true_labels_np.squeeze()
        if len(predicted_labels_np.shape) > 1 : predicted_labels_np.squeeze()
        
        # Compute the confusion matrix for the current batch
        batch_conf_matrix = confusion_matrix(true_labels_np, predicted_labels_np, labels=range(self.num_classes))
        
        # Convert numpy confusion matrix to a torch tensor and update the overall confusion matrix
        self.confusion_matrix += torch.tensor(batch_conf_matrix, dtype=torch.int32)

    def reset(self):
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int32)

    def get_confusion_matrix(self):
        return self.confusion_matrix

class DataModule():

    def __init__(self, root='./data/', num_workers=8) -> None:
        
        if not os.path.exists(root):
            os.mkdir(root)
        self.root = root
        self.num_workers = num_workers

    
    def get_dataloader(self, train=True):
        pass


    def train_dataloader(self):
        return self.get_dataloader(train=True)
    

    def val_dataloader(self):
        return self.get_dataloader(train=False)


class SimpleListDataset(Dataset):
    
    def __init__(self, data_list:list) -> None:
        super().__init__()
        self.data_list = data_list
        
    def __getitem__(self, index) -> Any:
        return self.data_list[index]
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    
def init_constant(module, const):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, const)
        nn.init.zeros_(module.bias)
    if type(module) == nn.Conv2d:
        nn.init.constant_(module.weight, const)
        nn.init.zeros_(module.bias)

def init_uniform(module, l=0, u=1):
    if type(module) == nn.Linear:
        nn.init.uniform_(module.weight, a=l, b=u)
        nn.init.zeros_(module.bias)
    if type(module) == nn.Conv2d:
        nn.init.uniform_(module.weight, a=l, b=u)
        nn.init.zeros_(module.bias)

def init_normal(module, mean=0, std=0.01):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=mean, std=std)
        nn.init.zeros_(module.bias)
    if type(module) == nn.Conv2d:
        nn.init.normal_(module.weight, mean=mean, std=std)
        nn.init.zeros_(module.bias)

def init_xavier_uniform(module, gain=1):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)
    if type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)
    if type(module) == nn.GRU:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])

def init_xavier_normal(module, gain=1):
    if type(module) == nn.Linear:
        nn.init.xavier_normal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)
    if type(module) == nn.Conv2d:
        nn.init.xavier_normal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)
    if type(module) == nn.GRU:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_normal_(module._parameters[param])
    

def lazy_layer_initialization(model, dummy_input, init_method=None):
    model(*dummy_input)
    if init_method is not None:
        model.apply(init_method)


def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor 
    # 1D valid_lens specifies valid length for each batch However the 2d valid_lens specifies valid lengths also 
    # for each sample in each batch aswell meaning samples in the same batch can have different lengths.
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X

    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)




def calculate_ouput_dim(net, input_dim=None):
    
    batch_size = 8
    for child in net.children():
        if isinstance(child, nn.Sequential):
            child = child[0]
        for layer in child.modules():
            if isinstance(layer, nn.Sequential):
                layer = layer[0]
            
            if isinstance(layer, nn.Linear):
                (W_1, in_size) = list(net.parameters())[0].shape
                dummy_input = torch.randn((batch_size, in_size))
                output = net(dummy_input)
            elif isinstance(layer, nn.Conv2d):
                (K, in_C, K_h, K_w) = list(net.parameters())[0].shape
                if input_dim == None:
                    dummy_input = torch.randn(batch_size, in_C, 1024, 1024)
                    print('Since the input dimension was not specified the default input size was set to be (1024, 1024)')
                else:
                    if input_dim[0] != in_C:
                        error_str = 'input channels {} doesnt match the input channel size of network {}'.format(input_dim[0], in_C)
                        raise Exception(error_str)
                    dummy_input = torch.randn(batch_size, in_C, input_dim[1], input_dim[2])
                
                output = net(dummy_input)
            break
        break
    
    return output.shape


    
    