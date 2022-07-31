import os
import torch
from torch import nn

from consts import CLS, SEP  # for bert
from data_load import tokenizer, trigger2idx, all_triggers, all_arguments
from model import Net

def predict_input(model, sentence):
    # model.eval()
    words = sentence.split()
    sentence_split, tokens_x, is_heads = [], [], []
    sentence_split.extend([CLS] + words + [SEP])
    for w in sentence_split:
        tokens = tokenizer.tokenize(w) if w not in [CLS, SEP] else [w]
        tokens_xx = tokenizer.convert_tokens_to_ids(tokens)

        if w in [CLS, SEP]:
            is_head = [0]
        else:
            is_head = [1] + [0] * (len(tokens) - 1)
        tokens_x.extend(tokens_xx), is_heads.extend(is_head)
    head_indexes = []
    for i in range(len(is_heads)):
        if is_heads[i]:
            head_indexes.append(i)
    max_len = 50
    tokens_x_2d = []
    tmp = tokens_x + [0] * (max_len - len(tokens_x))
    tokens_x_2d.append(tmp)
    head_indexes_2d = head_indexes + [0] * (max_len - len(head_indexes))

    trigger_pred = model.module.predict_triggers_input(words, tokens_x_2d, head_indexes_2d)
    print(trigger_pred)


os.environ["CUDA_VISIBLE_DEVICES"] = '3'

# 载入训练好的模型
if __name__ == "__main__":
    best_model_path = 'best_model.pt'
    # sentence = 'McCaleb , a construction foreman , had just turned 51 when he was killed on Sept . 22, 1985 .'
    sentence = input("Please enter a sentence,and we will get its triggers and arguments: ")

    "正确的torch.load方式"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_model = Net(
        device=device,
        trigger_size=len(all_triggers),
        argument_size=len(all_arguments)
    )
    best_model = nn.DataParallel(best_model)
    checkpoint = torch.load(best_model_path)
    best_model.load_state_dict(checkpoint['model'])
    best_model.eval()
    # if device == 'cuda':
    #     best_model = best_model.cuda()

    predict_input(best_model, sentence)
