# -*- coding: utf-8 -*-
# @Time    : 2022-05-03 17:42
# @Author  : 吴佳杨

import paddle
from paddlenlp.transformers import UnifiedTransformerTokenizer
from utils import MemoryDataset, RoFormerTokenizerForDialogue
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time

batch_size = 32
dataset_path = 'data/eval.txt'
tokenizer_name = 'plato-mini'
load_from = 'plato-mini_model'
save_to = 'experiment_result/transformer_eval.csv'
device = 'gpu'
paddle.device.set_device(device)


@paddle.no_grad()
def profile(mode='sformer'):
    assert mode in ['sformer', 'transformer', 'roformer']
    if mode == 'transformer':
        from paddlenlp.transformers import UnifiedTransformerLMHeadModel
        model = UnifiedTransformerLMHeadModel.from_pretrained(tokenizer_name)
        tokenizer = UnifiedTransformerTokenizer.from_pretrained(tokenizer_name)
        state = None
        dataset = MemoryDataset(tokenizer, dataset_path, mode=mode, batch_size=1)
    elif mode == 'roformer':
        from paddlenlp.transformers import RoFormerForPretraining
        model = RoFormerForPretraining.from_pretrained('roformer-chinese-base')
        tokenizer = RoFormerTokenizerForDialogue.from_pretrained('roformer-chinese-base')
        state = None
        dataset = MemoryDataset(tokenizer, dataset_path, mode='transformer', batch_size=1)
    else:
        from model import UnifiedStateTransformerLMHeadModel
        model = UnifiedStateTransformerLMHeadModel.from_pretrained(load_from)
        tokenizer = UnifiedTransformerTokenizer.from_pretrained(tokenizer_name)
        state = paddle.load(load_from + '/init_state.pdtensor')  # shape = (batch_size, state_size, hidden_size)
        print('\nInitial state loading succeeded! State size is %d\n' % state.shape[1])
        dataset = MemoryDataset(tokenizer, dataset_path, mode=mode, batch_size=1)
    input_lengths = []
    compute_times = []
    for idx, ds in enumerate(tqdm(dataset)):
        if device == 'cpu' and idx >= 10:
            break
        enc_inputs = ds['encoded_inputs']
        for enc_input in enc_inputs:
            input_lengths.append(enc_input['input_ids'].shape[1])
            if mode == 'sformer':
                enc_input['state'] = state
            if mode == 'roformer':
                del enc_input['position_ids']
            t = time()
            model(**enc_input)
            compute_times.append(time() - t)
    df = pd.DataFrame({'input_length': input_lengths, 'compute_time': compute_times})
    save_name = 'experiment_result\%s_s-%d' % (mode, state.shape[1]) if mode == 'sformer' else 'experiment_result\%s' % mode
    save_name = save_name + '_cpu' if device == 'cpu' else save_name
    save_name = save_name + '.csv'
    df.to_csv(save_name, index=False)
    print('Total %d inputs' % len(input_lengths))
    print('Average input length is %f' % (sum(input_lengths) / len(input_lengths)))
    print('Average compute time is %f' % (sum(compute_times) / len(compute_times)))


def plot():
    transformer = pd.read_csv('experiment_result/transformer_cpu.csv')
    transformer = [sum(transformer.loc[i::10, 'compute_time']) / len(transformer.loc[i::10, 'compute_time'])
                   for i in range(10)]
    roformer_small = pd.read_csv('experiment_result/roformer_small_cpu.csv')
    roformer_small = [sum(roformer_small.loc[i::10, 'compute_time']) / len(roformer_small.loc[i::10, 'compute_time'])
                      for i in range(10)]
    roformer_base = pd.read_csv('experiment_result/roformer_cpu.csv')
    roformer_base = [sum(roformer_base.loc[i::10, 'compute_time']) / len(roformer_base.loc[i::10, 'compute_time'])
                      for i in range(10)]
    sformer_s4 = pd.read_csv('experiment_result/sformer_s-4_cpu.csv')
    sformer_s4 = [sum(sformer_s4.loc[i::10, 'compute_time']) / len(sformer_s4.loc[i::10, 'compute_time'])
                  for i in range(10)]
    sformer_s8 = pd.read_csv('experiment_result/sformer_s-8_cpu.csv')
    sformer_s8 = [sum(sformer_s8.loc[i::10, 'compute_time']) / len(sformer_s8.loc[i::10, 'compute_time'])
                  for i in range(10)]
    sformer_s16 = pd.read_csv('experiment_result/sformer_s-16_cpu.csv')
    sformer_s16 = [sum(sformer_s16.loc[i::10, 'compute_time']) / len(sformer_s16.loc[i::10, 'compute_time'])
                   for i in range(10)]
    plt.plot(transformer, label='Plato-mini')
    plt.plot(roformer_small, label='RoFormer-small')
    plt.plot(roformer_base, label='RoFormer-base')
    plt.plot(sformer_s4, label='SFormer S-4')
    plt.plot(sformer_s8, label='SFormer S-8')
    plt.plot(sformer_s16, label='SFormer S-16')
    plt.xticks(list(range(10)))
    plt.legend()
    plt.xlabel("Dialogue Round")
    plt.ylabel("Timecount(s)")
    plt.show()


@paddle.no_grad()
def eval():
    assert 'memory' not in load_from
    if 'plato' in load_from:
        from paddlenlp.transformers import UnifiedTransformerLMHeadModel
        model = UnifiedTransformerLMHeadModel.from_pretrained(load_from)
        tokenizer = UnifiedTransformerTokenizer.from_pretrained(tokenizer_name)
    else:
        from paddlenlp.transformers import RoFormerForPretraining
        model = RoFormerForPretraining.from_pretrained(load_from)
        tokenizer = RoFormerTokenizerForDialogue.from_pretrained(tokenizer_name)
    dataset = MemoryDataset(tokenizer, dataset_path, mode='transformer', batch_size=batch_size, max_epoch=1)
    dataloader = paddle.io.DataLoader(dataset, return_list=True, batch_size=None, num_workers=4)

    def compute(data):
        inputs = data['encoded_inputs']
        distances = data['distances']
        ask_idxs = data['ask_idxs']
        total = distances.shape[0]  # batch_size
        total_distance = paddle.sum(distances)
        correct_mask = paddle.zeros_like(distances, dtype=paddle.float32)  # 1为回答正确，x[mask]中paddle不支持x和mask同为bool
        for i, model_input in enumerate(inputs):
            if paddle.any(ask_idxs == i):  # 根据answer评估记忆力
                if 'plato' in load_from:
                    output = model(**model_input)
                else:
                    del model_input['position_ids']
                    output, _ = model(**model_input)
                y_true = paddle.zeros_like(model_input['input_ids'])
                y_true[:, :-1] = model_input['input_ids'][:, 1:]  # 错位
                answer_idxs = paddle.any(ask_idxs == i, axis=-1)
                answer_mask = paddle.cast(model_input['token_type_ids'][answer_idxs], dtype='bool')
                answer_true = y_true[answer_idxs]
                answer_pred = output[answer_idxs]
                answer_mask = paddle.logical_not(answer_mask)
                answer_pred = paddle.argmax(answer_pred, axis=-1)
                result = paddle.all(paddle.logical_or(paddle.equal(answer_pred, answer_true), answer_mask),
                                    axis=-1)  # shape=(n,)，True代表该样本全对
                correct_mask[ask_idxs == i] = paddle.cast(result, correct_mask.dtype)
        correct_mask = paddle.all(paddle.cast(correct_mask, 'bool'), axis=-1)
        correct = paddle.sum(paddle.cast(correct_mask, dtype=paddle.int32))  # paddle.sum不支持int8和16
        correct_mask = correct_mask.numpy()
        distances = distances.numpy()
        correct_distance = np.sum(distances[correct_mask])  # paddle.sum不支持[]的求和
        return {
            'correct': correct,
            'total': total,
            'correct_distance': correct_distance,
            'total_distance': total_distance
        }

    def evaluation(dataloader):
        total, correct, total_distance, correct_distance = 0, 0, 0, 0
        model.eval()
        for step, data in tqdm(enumerate(dataloader)):
            log = compute(data)
            total += log['total']
            correct += log['correct']
            total_distance += log['total_distance']
            correct_distance += log['correct_distance']
        acc = correct / (total + 1e-9)
        mem = correct_distance / (total_distance + 1e-9)
        log = 'acc: %f,\tmem: %f,\ttotal: %d\n' % (acc, mem, total)
        print(log)
        return {
            'acc': float(acc),
            'mem': float(mem),
            'total': total
        }

    df = []
    for max_epoch in range(1, 11):
        dataset.max_epoch = max_epoch
        print('max_epoch: %d' % max_epoch)
        df.append(evaluation(dataloader))
        df[-1]['max_epoch'] = max_epoch
    df = pd.DataFrame(df)
    df.to_csv(save_to, index=False)


def plot_eval():
    transformer = pd.read_csv('experiment_result/transformer_eval.csv')
    roformer_small = pd.read_csv('experiment_result/roformer_small_eval.csv')
    roformer_base = pd.read_csv('experiment_result/roformer_base_eval.csv')
    ax1 = plt.subplot(2, 1, 1)  # （行，列，活跃区）
    plt.plot(list(range(1, 11)), transformer['acc'], label='Plato-mini')
    plt.plot(list(range(1, 11)), roformer_small['acc'], label='RoFormer-small')
    plt.plot(list(range(1, 11)), roformer_base['acc'], label='RoFormer-base')
    plt.plot(list(range(1, 11)), [0.622] * 10, label='SFormer S-4')
    plt.plot(list(range(1, 11)), [0.649] * 10, label='SFormer S-8')
    plt.plot(list(range(1, 11)), [0.668] * 10, label='SFormer S-16')
    # plt.plot(list(range(1, 11)), [0.186] * 10, label='SFormer S-8 without pretraining')
    plt.xticks(list(range(1, 11)))
    plt.legend()
    plt.xlabel("Max Concatenating Rounds")
    plt.ylabel("Acc")

    ax2 = plt.subplot(2, 1, 2, sharex=ax1)  # （行，列，活跃区）
    plt.plot(list(range(1, 11)), transformer['mem'], label='Plato-mini')
    plt.plot(list(range(1, 11)), roformer_small['mem'], label='RoFormer-small')
    plt.plot(list(range(1, 11)), roformer_base['mem'], label='RoFormer-base')
    plt.plot(list(range(1, 11)), [0.608] * 10, label='SFormer S-4')
    plt.plot(list(range(1, 11)), [0.630] * 10, label='SFormer S-8')
    plt.plot(list(range(1, 11)), [0.651] * 10, label='SFormer S-16')
    # plt.plot(list(range(1, 11)), [0.155] * 10, label='SFormer S-8 without pretraining')
    plt.legend()
    plt.xlabel("Max Concatenating Rounds")
    plt.ylabel("Mem")
    plt.show()


# profile('roformer')
# plot()
# eval()
plot_eval()
