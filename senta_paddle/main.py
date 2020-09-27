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
from __future__ import print_function

import os
import time
import argparse
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
from paddle.framework import manual_seed
import nets
import reader
from utils import ArgumentGroup
from utils import get_cards

parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("checkpoints", str, "checkpoints", "Path to save checkpoints")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("epoch", int, 1, "Number of epoches for training.")
# train_g.add_arg("epoch", int, 50, "Number of epoches for training.")
# train_g.add_arg("save_steps", int, 200,
train_g.add_arg("save_steps", int, 400,
                "The steps interval to save checkpoints.")
# train_g.add_arg("validation_steps", int, 200,
train_g.add_arg("validation_steps", int, 100,
                "The steps interval to evaluate model performance.")
train_g.add_arg("lr", float, 0.01, "The Learning rate value for training.")
train_g.add_arg("padding_size", int, 512, # 150,
                "The padding size for input sequences.")

log_g = ArgumentGroup(parser, "logging", "logging related")
log_g.add_arg("skip_steps", int, 1, "The steps interval to print loss.")
log_g.add_arg("verbose", bool, False, "Whether to output verbose log")

data_g = ArgumentGroup(parser, "data",
                       "Data paths, vocab paths and data processing options")
data_g.add_arg("data_dir", str, "./senta_data/", "Path to training data.")
data_g.add_arg("vocab_path", str, "./senta_data/word_dict.txt",
               "Vocabulary path.")
data_g.add_arg("vocab_size", int, 33256, "Vocabulary path.")
data_g.add_arg("batch_size", int, 2, # 256,
               "Total examples' number in batch for training.")
data_g.add_arg("random_seed", int, 123, "Random seed.") # changed by liujiaqi

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda", bool, True, "If set, use GPU for training.")
run_type_g.add_arg("do_train", bool, True, "Whether to perform training.")
run_type_g.add_arg("do_val", bool, True, "Whether to perform evaluation.")
run_type_g.add_arg("do_infer", bool, False, "Whether to perform inference.")
run_type_g.add_arg("profile_steps", int, 60000,
                   "The steps interval to record the performance.")
train_g.add_arg("model_type", str, "cnn_net", "Model type of training.")
parser.add_argument("--ce", action="store_true", help="run ce")

args = parser.parse_args()
# FLAGS_selected_gpus=0,1,2,3,4,5,6,7
if args.use_cuda:
    place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', "2")))
    # place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', "0,1,2")))
    dev_count = fluid.core.get_cuda_device_count()
    print(dev_count)
else:
    place = fluid.CPUPlace()
    dev_count = 1

# os.environ['FLAGS_selected_gpus']= "7"

manual_seed(123)
 
args.random_seed = 123
args.ce = True
if args.ce:
    print("ce mode")
    seed = args.random_seed
    np.random.seed(seed)
    fluid.default_startup_program().random_seed = seed
    fluid.default_main_program().random_seed = seed


def train():
    # with fluid.dygraph.guard(place):
    with fluid.dygraph.guard():
        if args.ce:
            print("ce mode")
            seed = args.random_seed
            np.random.seed(seed)
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
        processor = reader.SentaProcessor(
            data_dir=args.data_dir,
            vocab_path=args.vocab_path,
            random_seed=args.random_seed)
        num_labels = len(processor.get_labels())

        if not args.ce:
            train_data_generator = processor.data_generator(
                batch_size=args.batch_size,
                phase='train',
                epoch=args.epoch,
                shuffle=True)

            eval_data_generator = processor.data_generator(
                batch_size=args.batch_size,
                phase='dev',
                epoch=args.epoch,
                shuffle=False)
        else:
            train_data_generator = processor.data_generator(
                batch_size=args.batch_size,
                phase='train',
                epoch= args.epoch,
                shuffle=False)

            eval_data_generator = processor.data_generator(
                batch_size=args.batch_size,
                phase='dev',
                epoch=args.epoch,
                shuffle=False)
 
 
        model = nets.CNN(args.vocab_size)
      
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
        for parameters in model.named_parameters():
            print(parameters[0])
            if 'embedding' in parameters[0]:
                print(model.state_dict()[parameters[0]][0].shape)

        # sgd_optimizer = fluid.optimizer.Adagrad(learning_rate=args.lr,parameter_list=model.parameters())
        sgd_optimizer = paddle.fluid.optimizer.SGD(learning_rate=args.lr,parameter_list=model.parameters())
        steps = 0
        total_cost, total_acc, total_num_seqs = [], [], []
        gru_hidden_data = np.zeros((args.batch_size, 128), dtype='float64')
        ce_time, ce_infor = [], []
        reader_time = 0.0
        
        num_train_examples = processor.get_num_examples(phase="train")

        for eop in range(args.epoch):
            time_begin = time.time()
            for batch_id, data in enumerate(train_data_generator()):
                reader_begin = time.time()
                seq_len_arr = np.array([len(x[0]) for x in data], dtype="int64")
                steps += 1
                seq_len = to_variable(seq_len_arr)
                doc = to_variable(
                    np.array([
                        np.pad(x[0][0:args.padding_size], (
                            0, args.padding_size - len(x[0][
                                0:args.padding_size])),
                                'constant',
                                constant_values=0)
                        for x in data
                    ]).astype('int64'))

                label = to_variable(
                    np.array([x[1] for x in data]).astype('int64').reshape(
                        args.batch_size, 1))

                reader_end = time.time()
                reader_time += (reader_end - reader_begin)
                model.train()          
     
                avg_cost, prediction, acc = model(doc, seq_len, args.padding_size, label) 
                model.clear_gradients()        
                avg_cost.backward()

                sgd_optimizer.minimize(avg_cost)

   
                # np_mask = (doc.numpy() != 0).astype('int32')
                # word_num = np.sum(np_mask)
                word_num = np.sum(seq_len_arr)

                total_cost.append(avg_cost.numpy() * word_num)
                total_acc.append(acc.numpy() * word_num)
                total_num_seqs.append(word_num)
                
                if steps % args.skip_steps == 0:
                    time_end = time.time()
                    used_time = time_end - time_begin
                    
                    print("step: %d, ave loss: %f, "
                            "ave acc: %f, speed: %f steps/s, reader speed: %f steps/s" %
                            (steps,
                            np.sum(total_cost) / np.sum(total_num_seqs),
                            np.sum(total_acc) / np.sum(total_num_seqs),
                            args.skip_steps / used_time,
                            args.skip_steps / reader_time))
                    reader_time = 0.0
                    ce_time.append(used_time)
                    ce_infor.append(np.sum(total_acc) / np.sum(total_num_seqs))
                    total_cost, total_acc, total_num_seqs = [], [], []
                    time_begin = time.time()
                    
                    # if steps % args.validation_steps == 0:
                    #     total_eval_cost, total_eval_acc, total_eval_num_seqs = [], [], []
                    #     model.eval()
                    #     eval_steps = 0
                    #     gru_hidden_data = np.zeros((args.batch_size, 128), dtype='float64')
                    #     for eval_batch_id, eval_data in enumerate(
                    #             eval_data_generator()):
                    #         eval_seq_arr = np.array([len(x[0]) for x in data], dtype="int64")
                    #         eval_seq_len = to_variable(eval_seq_arr)
                    #         eval_np_doc = np.array([
                    #             np.pad(x[0][0:args.padding_size],
                    #                    (0, args.padding_size -
                    #                     len(x[0][0:args.padding_size])),
                    #                    'constant',
                    #                    constant_values=0) # args.vocab_size))
                    #             for x in eval_data
                    #         ]).astype('int64')# .reshape(-1)
                    #         eval_label = to_variable(
                    #             np.array([x[1] for x in eval_data]).astype(
                    #                 'int64').reshape(args.batch_size, 1))
                    #         eval_doc = to_variable(eval_np_doc)
                    #         eval_avg_cost, eval_prediction, eval_acc = model(
                    #             eval_doc, eval_seq_len, args.padding_size, eval_label)
                    #         eval_np_mask = (
                    #             eval_np_doc != 0).astype('int32')
                    #             # eval_np_doc != args.vocab_size).astype('int32')
                    #         # eval_word_num = np.sum(eval_np_mask)
                    #         eval_word_num = np.sum(eval_seq_arr)
                    #         total_eval_cost.append(eval_avg_cost.numpy() *
                    #                                eval_word_num)
                    #         total_eval_acc.append(eval_acc.numpy() *
                    #                               eval_word_num)
                    #         total_eval_num_seqs.append(eval_word_num)

                    #         eval_steps += 1

                    #     time_end = time.time()
                    #     used_time = time_end - time_begin
                    #     print(
                    #         "Final validation result: step: %d, ave loss: %f, "
                    #         "ave acc: %f, speed: %f steps/s" %
                    #         (steps, np.sum(total_eval_cost) /
                    #          np.sum(total_eval_num_seqs), np.sum(total_eval_acc)
                    #          / np.sum(total_eval_num_seqs),
                    #          eval_steps / used_time))
                    #     time_begin = time.time()
                    #     if args.ce:
                    #         print("kpis\ttrain_loss\t%0.3f" %
                    #               (np.sum(total_eval_cost) /
                    #                np.sum(total_eval_num_seqs)))
                    #         print("kpis\ttrain_acc\t%0.3f" %
                    #               (np.sum(total_eval_acc) /
                    #                np.sum(total_eval_num_seqs)))
                    
                    # if steps % args.save_steps == 0:
                    #     save_path = args.checkpoints+"/"+"save_dir_" + str(steps)
                    #     print('save model to: ' + save_path)
                    #     fluid.dygraph.save_dygraph(model.state_dict(),
                    #                                save_path)
                        # fluid.dygraph.save_dygraph(model.state_dict(),
                                                  #  save_path)
        if args.ce:
            card_num = get_cards()
            _acc = 0
            _time = 0
            try:
                _time = ce_time[-1]
                _acc = ce_infor[-1]
            except:
                print("ce info error")
            print("kpis\ttrain_duration_card%s\t%s" % (card_num, _time))
            print("kpis\ttrain_acc_card%s\t%f" % (card_num, _acc))

def main():
    train()

if __name__ == '__main__':
    print(args)
    main()
