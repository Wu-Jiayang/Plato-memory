# -*- coding: utf-8 -*-
# @Time    : 2022-01-20 21:57
# @Author  : 吴佳杨

import paddle
from paddlenlp.transformers import RoFormerTokenizer
import random
import json
import tqdm
from typing import Dict
import numpy as np


class DialogueGenerater():
    def __init__(self, tokenizer, model, steps=1000, dialogue_epochs=10, ask_num=3):
        """
        Args:
            steps (int): 共生成几场对话
            dialogue_epochs (int): 每场对话共生成几轮，必须 >= 2 * ask_num
            ask_num (int): 每场对话的知识点数量，一个知识点对应一个tell、ask、answer
        """
        super(DialogueGenerater, self).__init__()
        assert (ask_num * 2) <= dialogue_epochs
        self.tokenizer = tokenizer
        self.model = model
        self.steps = steps
        self.dialogue_epochs = dialogue_epochs
        self.ask_num = ask_num

    def generate(self) -> Dict:
        """
        通过规则生成tell、ask、answer，并以此为骨干，利用plato-mini生成中文对话
        """
        idx = random.sample(list(range(self.dialogue_epochs)), self.ask_num * 2)
        idx = np.sort(np.reshape(idx, (-1, 2)))
        tell_idxs = idx[:, 0]
        ask_idxs = idx[:, 1]
        distances = ask_idxs - tell_idxs
        memory_types = random.sample(['name', 'age', 'birthday', 'profession', 'hobby', 'hometown'], self.ask_num)
        memory_dialogues = [getattr(self, '_' + memory_type)() for memory_type in memory_types]  # [{'tell': ..., 'ask': ..., 'answer': ...}, ...]
        dialogues = []
        tell_idx = np.argwhere(tell_idxs == 0)      # 二维数组
        if tell_idx.shape[0] > 0:
            dialogues.append(random.choice(['', '你好！']) + memory_dialogues[tell_idx[0, 0]]['tell'])
        while len(dialogues) < self.dialogue_epochs * 2:
            tell_idx = np.argwhere(len(dialogues) == tell_idxs * 2)  # 二维数组
            ask_idx = np.argwhere(len(dialogues) == ask_idxs * 2)
            if tell_idx.shape[0] > 0:
                dialogues.append(memory_dialogues[tell_idx[0, 0]]['tell'])
            elif ask_idx.shape[0] > 0:
                dialogues.append(memory_dialogues[ask_idx[0, 0]]['ask'])
                dialogues.append(memory_dialogues[ask_idx[0, 0]]['answer'])
                continue
            # 调用dialogue_encode方法生成输入
            encoded_input = self.tokenizer.dialogue_encode(
                dialogues[-6:],
                add_start_token_as_response=True,
                return_tensors=True,
                is_split_into_words=False
            )
            encoded_input['input_ids'] = paddle.cast(encoded_input['input_ids'], 'int64')
            ids, scores = self.model.generate(
                **encoded_input,
                max_length=32,
                decode_strategy='sampling',
                top_k=5,
                num_return_sequences=1
            )
            ids = ids.numpy().tolist()[0]
            tokens = self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
            response = self.tokenizer.convert_tokens_to_string(tokens, keep_space=False)
            dialogues.append(response)
        return {'dialogues': dialogues, 'tell_idxs': tell_idxs.tolist(), 'ask_idxs': ask_idxs.tolist(), 'distances': distances.tolist()}

    def save_to_txt(self, file_name):
        """
        数据为随机生成

        Args:
            file_name (str): 保存的文件名
        """
        data = [json.dumps(self.generate(), ensure_ascii=False) for i in tqdm.tqdm(range(self.steps))]
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write('\n'.join(data))

    def _name(self):
        memory_dialogues = {}
        with open('data/name.txt', 'r', encoding='utf-8') as f:
            names = f.read().strip().split('\n')
        name = random.choice(names)
        memory_dialogues['tell'] = '我' + \
                                   random.choice(['', '的名字']) + \
                                   random.choice(['叫', '是']) + \
                                   name
        ask = random.choice(['我是谁', '我叫什么名字', '我的名字是什么'])
        if random.random() >= 0.5:
            ask = '还记得' + ask + '吗'
        memory_dialogues['ask'] = ask + '？'
        memory_dialogues['answer'] = name
        return memory_dialogues

    def _age(self):
        memory_dialogues = {}
        age = random.randint(3, 80)
        memory_dialogues['tell'] = '我' + \
                                   random.choice(['', '今年']) + \
                                   random.choice(['', '已经', '快', '差不多']) + \
                                   str(age) + \
                                   random.choice(['', '岁']) + \
                                   '了'
        ask = random.choice(['我多大岁数', '我今年多大', '我几岁了', '我的年龄是多少'])
        if random.random() >= 0.5:
            ask = '还记得' + ask + '吗'
        memory_dialogues['ask'] = ask + '？'
        memory_dialogues['answer'] = str(age) + '岁'
        return memory_dialogues

    def _hobby(self):
        memory_dialogues = {}
        with open('data/foods.txt', 'r', encoding='utf-8') as f:
            foods = f.read().strip().split('\n')
        with open('data/sports.txt', 'r', encoding='utf-8') as f:
            sports = f.read().strip().split('\n')
        with open('data/stars.txt', 'r', encoding='utf-8') as f:
            stars = f.read().strip().split('\n')
        hobby = random.choice([random.choice(foods), random.choice(sports), random.choice(stars)])
        ask = random.choice(['我喜欢什么', '我有什么爱好', '我痴迷于什么', "我热爱干啥"])
        if hobby[-1] == '1':  # 表示着是体育运动
            hobby = hobby[:-2]

        if hobby[-1] == '2':  # 表示着是食物
            hobby = '吃' + hobby[:-2]
            ask = random.choice(['我最爱吃什么', '我喜欢的美食是什么', '我的爱好是吃啥', "我喜欢吃啥"])

        if hobby[-1] == '3':  # 明星
            hobby = hobby[:-2]
            ask = random.choice(['我最爱哪个明星', '我喜欢的明星是谁', '我追哪位明星', "我喜欢哪位明星"])

        memory_dialogues['tell'] = '我' + \
                                   random.choice(['', '特别', '十分', '很', '非常', '尤其']) + \
                                   random.choice(['喜爱', '热爱', '喜欢', '爱', '痴迷于', '沉迷于']) + \
                                   hobby
        if random.random() >= 0.5:
            ask = '你还记得' + ask + '吗'
        memory_dialogues['ask'] = ask + '？'
        memory_dialogues['answer'] = random.choice([hobby[1:], hobby]) if hobby[0] == '吃' \
                                                                          or hobby[0] == "学" \
                                                                          or hobby[0] == '打' \
                                                                          or hobby[0] == '下' else hobby
        return memory_dialogues

    def _profession(self):
        memory_dialogues = {}
        with open('data/profession.txt', 'r', encoding='utf-8') as f:
            professions = f.read().strip().split('\n')
        profession = random.choice(professions)
        memory_dialogues['tell'] = '我' + \
                                   random.choice(['是', '是一名', '的职业是', '是个']) + \
                                   profession
        ask = random.choice(['我是干什么的', '我是做什么的', '我的职业是什么', "我是干啥的"])
        if random.random() >= 0.5:
            ask = '你还记得' + ask + '吗'
        memory_dialogues['ask'] = ask + '？'
        memory_dialogues['answer'] = profession
        return memory_dialogues

    def _birthday(self):
        memory_dialogues = {}
        year = random.randint(1920, 2018)
        month = random.randint(1, 12)
        maxday = 31 if month in [1, 3, 5, 7, 8, 10, 12] else 28 if month == 2 else 30
        day = random.randint(1, maxday)
        birth = random.choice(['%s年%s月%s日' % (year, month, day),
                               '%s-%s-%s' % (year, month, day),
                               '%s月%s号' % (month, day)
                               ])
        pre_word = random.choice(['生日是', '出生在', '出生于', '的生日在', ''])
        memory_dialogues['tell'] = '我' + \
                                   pre_word + birth + \
                                   (random.choice(['过生日', '生日', '出生']) if pre_word == '' else '')
        ask = random.choice(['我啥时候出生的',
                             '我生日啥时候',
                             '我啥时候过生日',
                             '我的生日是什么时候',
                             '我出生年月日是啥时候',
                             "我出生在哪一天",
                             "我诞生在哪一天",
                             "我母亲啥时候生的我"])
        if random.random() >= 0.5:
            ask = '你还记得' + ask + '吗'
        memory_dialogues['ask'] = ask + '？'
        memory_dialogues['answer'] = birth
        return memory_dialogues

    def _hometown(self):
        memory_dialogues = {}
        with open("data/Province.txt", 'r', encoding='utf-8') as file:
            Provinces = file.read().strip().split('\n')
            with open('data/hometown.txt', 'r', encoding='utf-8') as f:
                hometowns = f.read().strip().split('\n')
            Province = random.choice(Provinces)
            hometown = '' if random.random() > .2 else random.choice(hometowns)

            if "自治县" not in Province:
                Province = Province.replace('县', '') if random.random() > .5 else Province
            if "自治区" not in Province and "行政区" not in Province:
                Province = Province.replace('区', '') if random.random() > .5 else Province

            Province = Province.replace('省', '') if random.random() > .5 else Province
            Province = Province.replace('市', '') if random.random() > .5 else Province
            Province = Province.replace('特别行政区', '') if random.random() > .5 else Province
            Province = Province.replace('壮族自治区', '') if random.random() > .5 else Province
            hometown = Province + hometown
            verb = random.choice(['是', '来自'])
            memory_dialogues['tell'] = '我' + \
                                       verb + \
                                       hometown + \
                                       (random.choice(['人', '的']) if verb == '是' else '')
            ask = random.choice(['我是哪里人', '我来自哪里'])
            memory_dialogues['answer'] = hometown + '人' if ask == '我是哪里人' else hometown
            if random.random() >= 0.5:
                ask = '还记得' + ask + '吗'
            memory_dialogues['ask'] = ask + '？'
        return memory_dialogues

    def pad_batch_data(self, batch):
        """Pad the instances to the max sequence length in batch. """
        max_len = max(map(len, batch))
        batch_data = paddle.to_tensor(
            [
                list(data) + [0] * (max_len - len(data))
                for data in batch
            ],
            dtype='int64')
        return batch_data


class MemoryDataset(paddle.io.Dataset):
    """
    max_epoch: 最大历史对话轮数
    """
    def __init__(self, tokenizer, file_name: str, mode='sformer', batch_size=2, max_epoch=None):
        super(MemoryDataset, self).__init__()
        self.tokenizer = tokenizer
        self.mode = mode
        assert mode in ['sformer', 'transformer']
        assert max_epoch is None or max_epoch > 0
        with open(file_name, 'r', encoding='utf-8') as f:
            data = f.read().strip().split('\n')
            data = [json.loads(line) for line in data]
        self.data = data
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.shuffle()

    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))
        # return self.steps

    def shuffle(self):
        random.shuffle(self.data)

    def __getitem__(self, item) -> Dict:
        data = self.data[item * self.batch_size: (item + 1) * self.batch_size]
        dialogue_epochs = int(len(data[0]['dialogues']) / 2)
        for d in data:
            assert int(len(d['dialogues']) / 2) == dialogue_epochs
        distances = paddle.to_tensor([i['distances'] for i in data])
        ask_idxs = paddle.to_tensor([i['ask_idxs'] for i in data])
        encoded_inputs = []     # inputs for Response Generation
        state_update_inputs = []    # inputs for State Update
        for i in range(dialogue_epochs):
            input_ids = []
            token_type_ids = []
            position_ids = []
            attention_mask_temp = []
            for d in data:  # batch
                history = d['dialogues'][i * 2] if self.mode == 'sformer' else\
                    d['dialogues'][: i * 2 + 1] if self.max_epoch is None or i - self.max_epoch + 1 < 0 else\
                    d['dialogues'][(i - self.max_epoch + 1) * 2: i * 2 + 1]
                encoded_input_temp = self.tokenizer.dialogue_encode(
                    history,
                    response=d['dialogues'][i * 2 + 1],
                    return_tensors=False,
                    is_split_into_words=False
                )
                input_ids.append(encoded_input_temp['input_ids'])
                token_type_ids.append(encoded_input_temp['token_type_ids'])
                position_ids.append(encoded_input_temp['position_ids'])
                attention_mask_temp.append(encoded_input_temp['attention_mask'])     # numpy, shape=(seq_len, seq_len)
            input_ids = self.pad_batch_data(input_ids)
            token_type_ids = self.pad_batch_data(token_type_ids)
            position_ids = self.pad_batch_data(position_ids)
            attention_mask = np.zeros((input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1]))  # mask for Response Generation
            attention_mask2 = np.zeros_like(attention_mask)     # mask for State Update
            for j in range(input_ids.shape[0]):
                raw_seq_len = attention_mask_temp[j].shape[0]
                attention_mask[j, 0, :raw_seq_len, :raw_seq_len] = attention_mask_temp[j]
                attention_mask[j, ..., raw_seq_len:] = -1e9
                attention_mask2[j, ..., raw_seq_len:] = -1e9
            attention_mask = paddle.to_tensor(attention_mask, dtype=paddle.float32)
            attention_mask2 = paddle.to_tensor(attention_mask2, dtype=paddle.float32)
            encoded_inputs.append({
                'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'position_ids': position_ids,
                'attention_mask': attention_mask
            })
            state_update_inputs.append({
                'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'position_ids': position_ids,
                'attention_mask': attention_mask2
            })
        return {
            'encoded_inputs': encoded_inputs,
            'state_update_inputs': state_update_inputs,
            'distances': distances,
            'ask_idxs': ask_idxs
        }

    def pad_batch_data(self, batch):
        """Pad the instances to the max sequence length in batch. """
        max_len = max(map(len, batch))
        batch_data = paddle.to_tensor(
            [
                list(data) + [0] * (max_len - len(data))
                for data in batch
            ],
            dtype='int64')
        return batch_data


class RoFormerTokenizerForDialogue(RoFormerTokenizer):
    def dialogue_encode(self,
                        history,
                        response=None,
                        knowledge=None,
                        task_type=None,
                        max_seq_len=512,
                        max_response_len=128,
                        max_knowledge_len=128,
                        return_position_ids=True,
                        return_token_type_ids=True,
                        return_attention_mask=True,
                        return_length=False,
                        add_start_token_as_response=False,
                        pad_to_max_seq_len=False,
                        return_tensors=False,
                        is_split_into_words=True):
        """
        Main method to encode the single-turn or multi-turn dialogue conversation.
        It will return a dictionary containing the encoded sequence and other
        relative informations which meets the input format requirements of the
        UnifiedTransformer model.
        See detail at
        https://github.com/PaddlePaddle/Knover/tree/luge-dialogue/luge-dialogue

        Args:
            history (str|list|tuple): The history of dialogue conversation. It
                is an utterance or list of utterances to be encoded. Each
                utterance is a string.
            response (str, optional): The response of dialogue conversation.
                It should be set when training the model. It should not be set
                when running inference. Defaults to None.
            knowledge (str, optional): The knowledge information of dialogue
                conversation. It should be set if the `task_type` is "knowledge"
                or "recommend". Defaults to None.
            task_type (str, optional): The type of dialogue conversation. It is
                one of "chitchat", "knowledge" and "recommend". They represent
                the chitchat dialogue, knowledge grounded dialogue and
                conversational recommendation respectively. Defaults to None,
                which means there is no `special_token` added in output sequence
                for identifying different conversation types.
            max_seq_len (int, optional): The maximum encoded sequence length.
                Defaults to 512.
            max_response_len (int, optional): The maximum encoded sequence
                length of the input `response`. Defaults to 128.
            max_knowledge_len (int, optional): The maximum encoded sequence
                length of the input `knowledge`. Defaults to 128.
            return_position_ids (bool, optional): Whether to return the
                position_ids. Defaults to True.
            return_token_type_ids (bool, optional): Whether to return the
                token_type_ids. Defaults to True.
            return_attention_mask (bool, optional): Whether to return the
                attention_mask. Defaults to True.
            return_length (bool, optional): Whether to return the length of the
                encoded sequence. Defaults to False.
            add_start_token_as_response (bool, optional): Whether to add the
                special token "[CLS]" at the end of sequence as the begining of
                the response when running inference to force the model to start
                generating response sequence. Defaults to False.
            pad_to_max_seq_len (bool, optional): Whether to pad the returned
                sequences to the `max_seq_len`. Note that, in this method,
                returned sequences will be padded on the left. Defaults to False.
            return_tensors (bool, optional): Whether to convert the returned
                sequences to Tensor. Defaults to False.
            is_split_into_words(bool, optinal): Whether or not the input text
                (`history`, `response` and `knowledge`) has been pretokenized.
                Defaults to True.

        Returns:
            dict: A dictionary containing the encoded sequence and other
            relative informations.

            With the corresponding fields:

            - input_ids (list[int]|Tensor):
                A list of indices of input tokens to be feed to UnifiedTransformer
                model. If `return_tensors` is True, it is a Tensor with shape
                [1, sequence_length] and data type 'int64'.
            - token_type_ids (list[int]|Tensor, optional):
                A list of segment token indices to indicate whether the token
                belongs to the dialogue response. If `return_tensors` is True,
                it is a Tensor with shape [1, sequence_length] and data type
                'int64'.
                Being returned when `return_token_type_ids` is set to True.
            - position_ids (list[int]|Tensor, optional):
                A list of The position indices. If `return_tensors` is True,
                it is a Tensor with shape [1, sequence_length] and data type
                'int64'.
                Being returned when `return_position_ids` is set to True.
            - attention_mask (numpy.ndarray|Tensor, optional):
                A numpy.ndarray to prevents attention to some unwanted positions,
                with shape [sequence_length, sequence_length] and data type
                'float32'. If `return_tensors` is True, it is a Tensor with shape
                [1, 1, sequence_length, sequence_length] and data type 'float32'.
                Being returned when `return_attention_mask` is set to True.
            - seq_len (int, optional):
                The actual length of the `input_ids`, excluding the pad token.
                Being returned when `return_length` is set to True.

        Example:
            .. code-block::

                from paddlenlp.transformers import UnifiedTransformerTokenizer

                tokenizer = UnifiedTransformerTokenizer.from_pretrained('plato-mini')

                inputs = tokenizer.dialogue_encode('我爱祖国')
                for key in inputs:
                    print(key + ':')
                    print(inputs[key])
                # input_ids: [1, 6, 25445, 26907, 25475, 2]
                # token_type_ids: [0, 0, 0, 0, 0, 0]
                # position_ids: [0, 1, 2, 3, 4, 5]
                # attention_mask: [[0. 0. 0. 0. 0. 0.]
                # [0. 0. 0. 0. 0. 0.]
                # [0. 0. 0. 0. 0. 0.]
                # [0. 0. 0. 0. 0. 0.]
                # [0. 0. 0. 0. 0. 0.]
                # [0. 0. 0. 0. 0. 0.]]
        """

        # Input type checking for clearer error
        assert isinstance(history, str) or (
            isinstance(history, (list, tuple)) and
            (len(history) == 0 or len(history) != 0 and
             isinstance(history[0], str))), (
                 "The input `history` must be with type `str` (single context) "
                 "or `List[str]` (multi-turn context). But received: {}".format(
                     history))
        assert response is None or isinstance(response, str), (
            "The input `response` must of be with type `str`. But received: {}".
            format(response))
        assert knowledge is None or isinstance(knowledge, str), (
            "The input `knowledge` must of be with type `str`. But received: {}".
            format(knowledge))
        assert task_type is None or task_type in self.TASK_TO_SPECIAL_TOKEN, (
            "The input `task_type` must be None or one of {}.".format(", ".join(
                self.TASK_TO_SPECIAL_TOKEN.keys())))
        assert max_seq_len > max_response_len + max_knowledge_len, (
            "`max_seq_len` must be greater than the sum of `max_response_len` "
            "and `max_knowledge_len`. But received `max_seq_len` is {}, "
            "`max_response_len` is {}, `max_knowledge_len` is {}.".format(
                max_seq_len, max_response_len, max_knowledge_len))
        assert response is None or not add_start_token_as_response, (
            "`add_start_token_as_response` only works when `response` is "
            "`None`. But received `add_start_token_as_response`: `{}`, "
            "`response`: {}.".format(add_start_token_as_response, response))

        knowledge_ids = []
        if knowledge is not None:
            tokens = self._tokenize(knowledge)
            knowledge_ids = self.convert_tokens_to_ids(tokens)
            if len(knowledge_ids) > max_knowledge_len - 1:
                knowledge_ids = knowledge_ids[:max_knowledge_len - 1]
            knowledge_ids += [self.sep_token_id]

        response_ids = []
        if response is not None:
            tokens = self._tokenize(response)
            response_ids = [self.cls_token_id] + self.convert_tokens_to_ids(
                tokens)
            if len(response_ids) > max_response_len - 1:
                response_ids = response_ids[:max_response_len - 1]
            response_ids += [self.sep_token_id]
        elif add_start_token_as_response:
            response_ids = [self.cls_token_id]

        if task_type is not None:
            special_token = self.TASK_TO_SPECIAL_TOKEN[task_type]
            assert special_token in self.vocab._token_to_idx, (
                "The vocab file should contain the special token corresponding "
                "to the task: {}.".format(task_type))
            special_token_id = self.vocab._token_to_idx[special_token]
            knowledge_ids = [self.cls_token_id, special_token_id
                             ] + knowledge_ids
        else:
            knowledge_ids = [self.cls_token_id] + knowledge_ids

        max_history_len = max_seq_len - len(knowledge_ids) - len(response_ids)
        if isinstance(history, str):
            history = [history]
        history_ids = []
        for i in range(len(history) - 1, -1, -1):
            tokens = self._tokenize(history[i])
            if len(history_ids) + len(tokens) + 1 > max_history_len:
                if i == len(history) - 1:
                    tokens = tokens[1 - max_history_len:]
                    history_ids = (self.convert_tokens_to_ids(tokens) +
                                   [self.sep_token_id])
                break
            history_ids = (self.convert_tokens_to_ids(tokens) +
                           [self.sep_token_id]) + history_ids

        history_ids = knowledge_ids + history_ids
        # Build output dictionnary
        encoded_inputs = {}
        encoded_inputs["input_ids"] = history_ids + response_ids
        # Check lengths
        sequence_length = len(encoded_inputs["input_ids"])
        assert sequence_length <= max_seq_len

        # Considering that the logits at the last time step in the API of
        # generative task are taken to generate the next token. In order to
        # avoid the last time step being a pad, so take padding on the left.
        pad_length = max_seq_len - sequence_length if pad_to_max_seq_len else 0
        if pad_length > 0:
            encoded_inputs["input_ids"] = [
                self.pad_token_id
            ] * pad_length + encoded_inputs["input_ids"]
        if return_tensors:
            # Add dimention for batch_size
            encoded_inputs["input_ids"] = paddle.to_tensor(encoded_inputs[
                "input_ids"]).unsqueeze(0)

        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = [0] * len(
                history_ids) + [1] * len(response_ids)
            if pad_length > 0:
                encoded_inputs["token_type_ids"] = [
                    self.pad_token_id
                ] * pad_length + encoded_inputs["token_type_ids"]
            if return_tensors:
                # Add dimention for batch_size
                encoded_inputs["token_type_ids"] = paddle.to_tensor(
                    encoded_inputs["token_type_ids"]).unsqueeze(0)

        if return_length:
            encoded_inputs["seq_len"] = sequence_length

        if return_position_ids:
            encoded_inputs["position_ids"] = list(range(sequence_length))
            if pad_length > 0:
                encoded_inputs["position_ids"] = [
                    self.pad_token_id
                ] * pad_length + encoded_inputs["position_ids"]
            if return_tensors:
                # Add dimention for batch_size
                encoded_inputs["position_ids"] = paddle.to_tensor(
                    encoded_inputs["position_ids"]).unsqueeze(0)

        if return_attention_mask:
            attention_mask = np.ones(
                (sequence_length, sequence_length), dtype='float32') * -1e9
            start = len(history_ids)
            end = sequence_length
            attention_mask[:end, :start] = 0.0
            # Generate the lower triangular matrix using the slice of matrix
            tmp = np.triu(
                np.ones(
                    [end - start, end - start], dtype='float32') * -1e9, 1)
            attention_mask[start:end, start:end] = tmp
            encoded_inputs["attention_mask"] = attention_mask
            if pad_length > 0:
                new_mask = np.ones(
                    (max_seq_len, max_seq_len), dtype='float32') * -1e9
                new_mask[-sequence_length:, -sequence_length:] = attention_mask
                encoded_inputs["attention_mask"] = new_mask
            if return_tensors:
                # Add dimentions for batch_size and num_heads
                encoded_inputs["attention_mask"] = paddle.to_tensor(
                    encoded_inputs["attention_mask"]).unsqueeze((0, 1))

        return encoded_inputs
