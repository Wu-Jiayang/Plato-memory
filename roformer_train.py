# -*- coding: utf-8 -*-
# @Time    : 2022-04-01 1:08
# @Author  : 吴佳杨

import paddle
from paddlenlp.transformers import RoFormerForPretraining, LinearDecayWithWarmup
from utils import MemoryDataset, RoFormerTokenizerForDialogue
from tqdm import tqdm
import numpy as np

tokenizer_name = 'roformer-chinese-small'
load_from = 'roformer-chinese-small'
save_to = 'roformer_small_model'
train_dataset = 'data/train.txt'
eval_dataset = 'data/eval.txt'
batch_size = 32     # aistudio可使用32
epochs = 30
num_workers = 4
init_lr = 1e-4

tokenizer = RoFormerTokenizerForDialogue.from_pretrained(tokenizer_name)
model = RoFormerForPretraining.from_pretrained(load_from)
# model.apply(model.init_weights)     # 使用初始化权重
model.save_pretrained(save_to)
train_dataset = MemoryDataset(tokenizer, train_dataset, mode='transformer', batch_size=batch_size)
train_dataloader = paddle.io.DataLoader(train_dataset, return_list=True, batch_size=None, num_workers=num_workers)
eval_dataset = MemoryDataset(tokenizer, eval_dataset, mode='transformer', batch_size=batch_size)
eval_dataloader = paddle.io.DataLoader(eval_dataset, return_list=True, batch_size=None, num_workers=num_workers)
total_step = len(train_dataset) * epochs * 10   # 每轮对话反向传播一次
loss_fn = paddle.nn.loss.CrossEntropyLoss(reduction='sum')
lr_scheduler = LinearDecayWithWarmup(init_lr, total_step, 0.1)
decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
opt = paddle.optimizer.AdamW(lr_scheduler, parameters=model.parameters(), apply_decay_param_fun=lambda x: x in decay_params)


def compute(data):
    total_loss = 0
    inputs = data['encoded_inputs']
    distances = data['distances']
    ask_idxs = data['ask_idxs']
    total = distances.shape[0]      # batch_size
    total_distance = paddle.sum(distances)
    correct_mask = paddle.zeros_like(distances, dtype=paddle.float32)     # 1为回答正确，x[mask]中paddle不支持x和mask同为bool
    for i, model_input in enumerate(inputs):
        if paddle.any(ask_idxs == i):   # 根据answer评估记忆力
            del model_input['position_ids']
            output, _ = model(**model_input)
            y_true = paddle.zeros_like(model_input['input_ids'])
            y_true[:, :-1] = model_input['input_ids'][:, 1:]  # 错位
            answer_idxs = paddle.any(ask_idxs == i, axis=-1)
            answer_mask = paddle.cast(model_input['token_type_ids'][answer_idxs], dtype='bool')
            answer_true = y_true[answer_idxs]
            answer_pred = output[answer_idxs]
            loss = loss_fn(answer_pred[answer_mask], answer_true[answer_mask])     # 仅训练记忆力
            total_loss += loss
            answer_mask = paddle.logical_not(answer_mask)
            answer_pred = paddle.argmax(answer_pred, axis=-1)
            result = paddle.all(paddle.logical_or(paddle.equal(answer_pred, answer_true), answer_mask),
                                axis=-1)  # shape=(n,)，True代表该样本全对
            correct_mask[ask_idxs == i] = paddle.cast(result, correct_mask.dtype)
            loss.backward()
            opt.step()
            lr_scheduler.step()
            opt.clear_grad()
    correct_mask = paddle.all(paddle.cast(correct_mask, 'bool'), axis=-1)
    correct = paddle.sum(paddle.cast(correct_mask, dtype=paddle.int32))  # paddle.sum不支持int8和16
    correct_mask = correct_mask.numpy()
    distances = distances.numpy()
    correct_distance = np.sum(distances[correct_mask])  # paddle.sum不支持[]的求和
    return {
        'loss': total_loss,
        'correct': correct,
        'total': total,
        'correct_distance': correct_distance,
        'total_distance': total_distance
    }


def train(epoch: int, dataloader):
    total, correct, total_distance, correct_distance, total_loss = 0, 0, 0, 0, 0
    model.train()
    for step, data in tqdm(enumerate(dataloader)):
        log = compute(data)
        total_loss += log['loss']
        total += log['total']
        correct += log['correct']
        total_distance += log['total_distance']
        correct_distance += log['correct_distance']
    acc = correct / (total + 1e-9)
    mem = correct_distance / (total_distance + 1e-9)
    loss = total_loss / (total + 1e-9)
    log = 'epoch: %d,\tlr: %f,\tloss: %f,\tacc: %f,\tmem: %f,\ttotal: %d\n' % (epoch, opt.get_lr(), loss, acc, mem, total)
    print(log)
    with open(save_to + '/log.txt', 'a', encoding='utf-8') as f:
        f.write(log)
    return {
        'loss': loss,
        'acc': acc,
        'mem': mem,
        'total': total
    }


@paddle.no_grad()
def evaluation(dataloader):
    total, correct, total_distance, correct_distance, total_loss = 0, 0, 0, 0, 0
    model.eval()
    for step, data in tqdm(enumerate(dataloader)):
        log = compute(data)
        total_loss += log['loss']
        total += log['total']
        correct += log['correct']
        total_distance += log['total_distance']
        correct_distance += log['correct_distance']
    acc = correct / (total + 1e-9)
    mem = correct_distance / (total_distance + 1e-9)
    loss = total_loss / (total + 1e-9)
    log = 'eval: {loss: %f,\tacc: %f,\tmem: %f,\ttotal: %d}\n' % (loss, acc, mem, total)
    print(log)
    with open(save_to + '/log.txt', 'a', encoding='utf-8') as f:
        f.write(log)
    return {
        'loss': loss,
        'acc': acc,
        'mem': mem,
        'total': total
    }


def do_train():
    best_acc = 0
    for i in range(epochs):
        train(i, train_dataloader)
        train_dataset.shuffle()
        log = evaluation(eval_dataloader)
        if log['acc'] >= best_acc:
            best_acc = log['acc']
            model.save_pretrained(save_to)
            with open(save_to + '/log.txt', 'a', encoding='utf-8') as f:
                f.write('Saved\n')


do_train()
