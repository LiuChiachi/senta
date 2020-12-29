# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear, Embedding
from paddle.fluid.dygraph import GRUUnit
from paddle.fluid.dygraph.base import to_variable
import numpy as np


class CNN(paddle.nn.Layer):
    def __init__(self, 
                 dict_dim, 
                 emb_dim=128,
                 hid_dim=128,
                 fc_hid_dim=96,
                 class_dim=2,
                 channels=1,
                 win_size=(3,128)):
        super(CNN, self).__init__()
        self.dict_dim = dict_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.fc_hid_dim = fc_hid_dim
        self.class_dim = class_dim
        self.channels = channels
        self.win_size = win_size
        self.embedding = Embedding(
            size=[self.dict_dim + 1, self.emb_dim],
            dtype='float64',
            is_sparse=False,
            padding_idx=0)
        self._conv2d = Conv2D(num_channels=self.channels,
            num_filters=self.hid_dim,
            filter_size=win_size,
            padding=[1,0],
            use_cudnn=True,
            act=None,
            dtype="float64")
        self._fc_1 = Linear(input_dim = self.hid_dim, output_dim=self.fc_hid_dim, dtype="float64")
        self._fc_2 = Linear(input_dim = self.fc_hid_dim,
                                 output_dim = self.class_dim,
                                 act="softmax",
                                 dtype="float64")
    def forward(self, inputs, seq_len, padding_size, label=None):
        # inputs的前后均被填充了1个时间步的0 [N, max_seq_len]
        emb = self.embedding(inputs) # 
        emb = fluid.layers.unsqueeze(input=emb, axes=[1])  # [N, 1, max_seq_len, emb_dim]
   
        # conv
        conv = self._conv2d(emb)  # [N, num_filters, max_seq_len, 1] 1是因为其实是1d的卷积
        conv = fluid.layers.tanh(conv)
        # 为了保证maxpooling的结果准确，需要在conv后、pooling前，把超过seq_len
        # 以下的时间步置为-INF
        mask = (paddle.fluid.layers.sequence_mask(seq_len, maxlen=padding_size, dtype='float64')-1)*1e6
        mask = fluid.layers.unsqueeze(input=mask, axes=[1, 3])
        conv_mask = mask + conv # [N, num_filters, max_seq_len, 1]

        # maxpooling
        pool = fluid.layers.reduce_max(conv_mask, dim=-2) # [N, num_filters,1]
        pool = fluid.layers.flatten(pool)  # [N, num_filters]

        fc_1 = self._fc_1(pool) # [N, fc_hid_dim]

        prediction = self._fc_2(fc_1) # [N, class_dim]

        if label is not None:
            cost = fluid.layers.cross_entropy(input=prediction, label=label)
            avg_cost = fluid.layers.mean(x=cost)
            acc = fluid.layers.accuracy(input=prediction, label=label)
            # import pdb; pdb.set_trace()
            return avg_cost, prediction, acc
        else:
            return prediction
