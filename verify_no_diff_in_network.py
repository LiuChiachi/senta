import numpy as np
# paddle import 
import os
import time
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
from paddle.framework import manual_seed

from senta_paddle import nets
from senta_paddle import reader

# torch import
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F


from senta_torch.model import TextCNN, SentimentCNN
from senta_torch.data import TextDataset

torch.manual_seed(123)
manual_seed(123)

data_dir = 'senta_paddle/senta_data'
vocab_path = "./senta_paddle/senta_data/word_dict.txt"
seed = 123
epoch = 1
batch_size = 2
vocab_size = 33256
padding_size = 512


def train_paddle_senta():
    paddle_loss_list = []
    with fluid.dygraph.guard():
        seed = 123
        np.random.seed(seed)
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed
        processor = reader.SentaProcessor(
        data_dir=data_dir,
        vocab_path=vocab_path,
        random_seed=seed)
        num_labels = len(processor.get_labels())

        train_data_generator = processor.data_generator(
            batch_size=batch_size,
            phase='train',
            epoch=epoch,
            shuffle=False)
 
        model = nets.CNN(vocab_size)
      
        # save initial param to files
        param_dict = {}
        for param_name in model.state_dict():
            param_dict[param_name] = model.state_dict()[param_name].numpy()
            if 'embedding' in param_name:
                state_dict = model.state_dict()
                param_dict[param_name][0] = 0
                state_dict[param_name] = paddle.to_variable(param_dict[param_name])
                model.set_dict(state_dict)
                # print(param_dict[param_name][0])
        np.savez('./paramters.npz', **param_dict)

        # sgd_optimizer = fluid.optimizer.Adagrad(learning_rate=args.lr,parameter_list=model.parameters())
        sgd_optimizer = paddle.fluid.optimizer.SGD(learning_rate=0.0,parameter_list=model.parameters())
        steps = 0
        gru_hidden_data = np.zeros((batch_size, 128), dtype='float64')

        num_train_examples = processor.get_num_examples(phase="train")

        for batch_id, data in enumerate(train_data_generator()):
            reader_begin = time.time()
            seq_len_arr = np.array([len(x[0]) for x in data], dtype="int64")
            steps += 1
            seq_len = to_variable(seq_len_arr)
            doc = to_variable(
                np.array([
                    np.pad(x[0][0: padding_size], (
                        0, padding_size - len(x[0][
                            0:padding_size])),
                            'constant',
                            constant_values=0)
                    for x in data
                ]).astype('int64'))

            label = to_variable(
                np.array([x[1] for x in data]).astype('int64').reshape(
                    batch_size, 1))

            model.train()            
            avg_cost, _, _ = model(doc, seq_len, padding_size, label)
            model.clear_gradients()        
            avg_cost.backward()
            sgd_optimizer.minimize(avg_cost)

            paddle_loss_list.append(avg_cost.numpy())
            if steps == 100:
                break
        return paddle_loss_list


def train_torch_senta():
    torch_loss_list = []
    training_set = TextDataset(path='senta_torch/data/train')
    import torch.utils.data as data
    training_iter = data.DataLoader(dataset=training_set,
                                    batch_size=batch_size,
                                    num_workers=2)
    model = SentimentCNN(vocab_size)
    param_dict = np.load('./paramters.npz', allow_pickle=True)
  
    for parameters in model.named_parameters():
        param_np = param_dict[parameters[0]]
        if 'weight' in parameters[0] and 'fc' in parameters[0]:
            param_np = param_np.transpose()
        if 'embedding' in parameters[0]:

            param_np[0] = 0
        parameters[1].data = torch.from_numpy(param_np)

    if torch.cuda.is_available():

        model = model.cuda()
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.SGD(model.parameters(), lr=0.0)
    steps = 0
 
    for data, label in training_iter:
        steps += 1
        seq_len = torch.from_numpy(np.array(data[:,-1:].reshape(batch_size)))
        doc = data[:,:-1]
        data_pad = [np.pad(x[0: padding_size], (0, padding_size-len(x[0: padding_size])), 'constant', constant_values=0)for x in data]
        doc = torch.from_numpy(np.array(data_pad).astype('int64'))

        label = torch.from_numpy(np.array([x for x in label]).astype('int64'))

        if torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()

        avg_cost, _, _ = model(data, seq_len, label)

        optimizer.zero_grad() # 清除梯度
        avg_cost.backward() # 计算梯度
        optimizer.step() # 更新参数
        torch_loss_list.append(avg_cost.data.cpu().numpy())
        if steps == 100:
            break
    return torch_loss_list

if __name__ == "__main__":
    paddle_loss_list = np.asarray(train_paddle_senta())
    torch_loss_list =  np.asarray(train_torch_senta())
    torch_loss_list = torch_loss_list.reshape(paddle_loss_list.shape)
    np.testing.assert_allclose(paddle_loss_list, torch_loss_list)