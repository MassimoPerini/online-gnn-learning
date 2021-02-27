import enum
import tensorflow as tf
from graphsage.pytorch.graphsage_dgl import GraphSAGE as GraphSAGE_pytorch
from graphsage.pytorch.model import *


class Lib_supported(enum.Enum):
    PYTORCH = 1
    TF = 2
    TF_STATIC = 3

LIB = None
_GPU_ID = None
_GPU = False

def init(library, GPU = False, GPU_ID = -1):
    global LIB
    global _GPU_ID
    global _GPU
    _GPU = GPU
    _GPU_ID = GPU_ID
    LIB = library
    if LIB == Lib_supported.TF:
        #import tensorflow as tf
        from graphsage.tf.model import RandomTfSupervisedGraphSage, PrioritizedTfSupervisedGraphSage, NoRehTfSupervisedGraphSage, FullTfSupervisedGraphSage
        from graphsage.tf.graphsage_dgl import GraphSAGE as GraphSAGE_tf
        return GraphSAGE_tf, RandomTfSupervisedGraphSage, PrioritizedTfSupervisedGraphSage, NoRehTfSupervisedGraphSage, FullTfSupervisedGraphSage, tf.keras.activations.relu
    elif LIB == Lib_supported.PYTORCH:
        import torch
        if _GPU and GPU_ID >= 0:
            torch.cuda.set_device(GPU)
        return GraphSAGE_pytorch, RandomPytorchSupervisedGraphSage, PrioritizedPytorchSupervisedGraphSage, NoRehPytorchSupervisedGraphSage, FullPytorchSupervisedGraphSage, torch.nn.functional.relu

    elif LIB == Lib_supported.TF_STATIC:
        #import tensorflow as tf
        from graphsage.tf_static.model import RandomTfSupervisedGraphSage, PrioritizedTfSupervisedGraphSage, NoRehTfSupervisedGraphSage, FullTfSupervisedGraphSage
        from graphsage.tf_static.graphsage_dgl import GraphSAGE as GraphSAGE_tf
        return GraphSAGE_tf, RandomTfSupervisedGraphSage, PrioritizedTfSupervisedGraphSage, NoRehTfSupervisedGraphSage, FullTfSupervisedGraphSage, tf.keras.activations.relu


def to_nn_lib(data, GPU = True, dtype = None):

    data_nn = None

    if LIB == Lib_supported.TF or LIB == Lib_supported.TF_STATIC:
        device_cpu = "/cpu:0"
        device_gpu = "/gpu:0"
        if _GPU_ID >= 0:
            device_gpu = "/gpu:"+str(_GPU_ID)
        if GPU:
            with tf.device(device_gpu):
                data_nn = tf.convert_to_tensor(data, dtype=dtype)
        else:
            with tf.device(device_cpu):
                data_nn = tf.convert_to_tensor(data, dtype=dtype)

    elif LIB == Lib_supported.PYTORCH:
        #if data.dtype == torch.long:
        #    data_nn = torch.LongTensor(data)
        #else:
        #    data_nn = torch.FloatTensor(data)
        data_nn = torch.tensor(data)
        if data_nn.dtype == torch.float64:
            data_nn = data_nn.float()
        if GPU:
            data_nn = data_nn.cuda()
    return data_nn

def from_nn_lib_to_list(data):
    if LIB == Lib_supported.TF or LIB == Lib_supported.TF_STATIC:
        return list(data.numpy())
    elif LIB == Lib_supported.PYTORCH:
        return data.tolist()

def from_nn_lib_to_set(data):
    if LIB == Lib_supported.TF or LIB == Lib_supported.TF_STATIC:
        return {v for v in data.numpy()}
    elif LIB == Lib_supported.PYTORCH:
        return {v.item() for v in data}

def from_nn_lib_to_numpy(data):
    if LIB == Lib_supported.TF or LIB == Lib_supported.TF_STATIC:
        return data.numpy()
    elif LIB == Lib_supported.PYTORCH:
        if data.is_cuda:
            return data.cpu().numpy()
        return data.numpy()

def from_nn_get_python_value(tensor):
    if LIB == Lib_supported.PYTORCH:
        return tensor.item()
    elif LIB == Lib_supported.TF or LIB == Lib_supported.TF_STATIC:
        return tf.keras.backend.get_value(tensor)

def create_nn_sparse(row, col, vals, size, GPU=True):
    if LIB == Lib_supported.PYTORCH:
        col_idx = np.array(col, copy=False, ndmin=2).reshape(1,-1)
        row_idx = np.array(row, copy=False, ndmin=2)
        index_matrix = torch.LongTensor(np.concatenate((row_idx, col_idx), axis=0))
        vals = torch.LongTensor(vals)
        matrix = torch.sparse.LongTensor(index_matrix, vals, torch.Size(size))
        if GPU:
            matrix = matrix.cuda()
        return matrix

    elif LIB == Lib_supported.TF or LIB == Lib_supported.TF_STATIC:
        index_matrix = np.asarray(list(zip(row, col)))
        device_cpu = "/cpu:0"
        device_gpu = "/gpu:0"
        if _GPU_ID >= 0:
            device_gpu = "/gpu:" + str(_GPU_ID)
        if GPU:
            with tf.device(device_gpu):
                matrix = tf.sparse.SparseTensor(indices=index_matrix, values=vals, dense_shape=size)
        else:
            with tf.device(device_cpu):
                matrix = tf.sparse.SparseTensor(indices=index_matrix, values=vals, dense_shape=size)
        return matrix

def get_context():
    if LIB == Lib_supported.PYTORCH:
        return torch.device('cuda:0')
    elif LIB == Lib_supported.TF_STATIC or LIB == Lib_supported.TF:
        return tf.device('/gpu:0')

def index_tensor(tensor, indices):
    if LIB == Lib_supported.PYTORCH:
        return tensor[indices]
    elif LIB == Lib_supported.TF_STATIC or LIB == Lib_supported.TF:
        return tf.gather(tensor, indices)

class sparse1d():
    def __init__(self, sparse_matrix):
        self.mtx = sparse_matrix

    def __getitem__(self, items):
        if hasattr(items, '__len__') and (not isinstance(items, str)):
            return np.squeeze(self.mtx[0, items].toarray())
        return self.mtx[0, items]

    def __setitem__(self, keys, items):
        self.mtx[0, keys] = items
