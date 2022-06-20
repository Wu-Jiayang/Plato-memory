import argparse
import os
from paddlenlp.transformers import UnifiedTransformerLMHeadModel, UnifiedTransformerTokenizer
from utils import DialogueGenerater


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('-o', '--output_dir', type=str, default='./output/', help='The dir of the output.')
    args = parser.parse_args()
    return args


def save_dataset(save_name: str, steps: int):
    model_name = 'plato-mini'
    tokenizer = UnifiedTransformerTokenizer.from_pretrained(model_name)
    model = UnifiedTransformerLMHeadModel.from_pretrained(model_name)
    dialogue_generater = DialogueGenerater(tokenizer, model, steps=steps)
    dialogue_generater.save_to_txt(os.path.join(args.output_dir, save_name))


def main(args):
    save_dataset('train.txt', steps=21000)
    save_dataset('eval.txt', steps=4000)


if __name__ == '__main__':
    args = parse_args()
    main(args)
