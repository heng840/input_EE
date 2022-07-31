import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from transformers import BertModel
from data_load import idx2trigger, argument2idx
from consts import NONE
from utils import find_triggers_sentence, find_triggers


class Net(nn.Module):
    def __init__(self, trigger_size=None, argument_size=None, device=torch.device("cpu")):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        hidden_size = 768
        self.fc1 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.ReLU(),
        )
        self.fc_trigger = nn.Sequential(
            nn.Linear(hidden_size, trigger_size),
        )
        self.fc_argument = nn.Sequential(
            nn.Linear(hidden_size * 2, argument_size),
        )
        self.device = device

    def predict_triggers(self, tokens_x_2d, head_indexes_2d, triggers_y_2d, arguments_2d):
        """
        :param tokens_x_2d:
        :param head_indexes_2d:
        :param triggers_y_2d:
        :param arguments_2d:
        :return:
        """
        tokens_x_2d = torch.LongTensor(tokens_x_2d).to(self.device)
        triggers_y_2d = torch.LongTensor(triggers_y_2d).to(self.device)
        head_indexes_2d = torch.LongTensor(head_indexes_2d).to(self.device)

        self.bert.train()
        encoded_layers = self.bert(tokens_x_2d)
        enc = encoded_layers[0]

        x = enc

        batch_size = tokens_x_2d.shape[0]

        for i in range(batch_size):
            x[i] = torch.index_select(x[i], 0, head_indexes_2d[i])

        trigger_logits = self.fc_trigger(x)
        trigger_hat_2d = trigger_logits.argmax(-1)
        # print(trigger_logits.shape, trigger_hat_2d.shape)

        argument_hidden, argument_keys = [], []
        for i in range(batch_size):
            candidates = arguments_2d[i]['candidates']
            golden_entity_tensors = {}

            for j in range(len(candidates)):
                e_start, e_end, e_type_str = candidates[j]
                golden_entity_tensors[candidates[j]] = x[i, e_start:e_end, ].mean(dim=0)

            predicted_triggers = find_triggers([idx2trigger[trigger] for trigger in trigger_hat_2d[i].tolist()])
            for predicted_trigger in predicted_triggers:
                t_start, t_end, t_type_str = predicted_trigger
                event_tensor = x[i, t_start:t_end, ].mean(dim=0)
                for j in range(len(candidates)):
                    e_start, e_end, e_type_str = candidates[j]
                    entity_tensor = golden_entity_tensors[candidates[j]]

                    argument_hidden.append(torch.cat([event_tensor, entity_tensor]))
                    argument_keys.append((i, t_start, t_end, t_type_str, e_start, e_end, e_type_str))

        return trigger_logits, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys

    def predict_arguments(self, argument_hidden, argument_keys, arguments_2d):
        argument_hidden = torch.stack(argument_hidden)
        argument_logits = self.fc_argument(argument_hidden)
        argument_hat_1d = argument_logits.argmax(-1)

        arguments_y_1d = []
        for i, t_start, t_end, t_type_str, e_start, e_end, e_type_str in argument_keys:
            a_label = argument2idx[NONE]
            if (t_start, t_end, t_type_str) in arguments_2d[i]['events']:
                for (a_start, a_end, a_type_idx) in arguments_2d[i]['events'][(t_start, t_end, t_type_str)]:
                    if e_start == a_start and e_end == a_end:
                        a_label = a_type_idx
                        break
            arguments_y_1d.append(a_label)

        arguments_y_1d = torch.LongTensor(arguments_y_1d).to(self.device)

        batch_size = len(arguments_2d)
        argument_hat_2d = [{'events': {}} for _ in range(batch_size)]
        for (i, st, ed, event_type_str, e_st, e_ed, entity_type), a_label in zip(argument_keys,
                                                                                 argument_hat_1d.cpu().numpy()):
            if a_label == argument2idx[NONE]:
                continue
            if (st, ed, event_type_str) not in argument_hat_2d[i]['events']:
                argument_hat_2d[i]['events'][(st, ed, event_type_str)] = []
            argument_hat_2d[i]['events'][(st, ed, event_type_str)].append((e_st, e_ed, a_label))

        return argument_logits, arguments_y_1d, argument_hat_1d, argument_hat_2d

    def predict_triggers_input(self, words, tokens_x_2d, head_indexes_2d):
        """
        :param tokens_x_2d:
        :param head_indexes_2d:
        :return:
        """
        tokens_x_2d = torch.LongTensor(tokens_x_2d).to(self.device)
        head_indexes_2d = torch.LongTensor(head_indexes_2d).to(self.device)

        self.bert.eval()
        with torch.no_grad():
            encoded_layers = self.bert(tokens_x_2d)
            enc = encoded_layers[0]
        x = enc

        trigger_logits = self.fc_trigger(x)  # 为什么是[1,50,68], 68???
        trigger_hat_2d = trigger_logits.argmax(-1)
        trigger_hat_2d = trigger_hat_2d.squeeze().tolist()
        # hat = max(trigger_hat_2d, key=trigger_hat_2d.count)
        hat = max(trigger_hat_2d)
        trigger_hat = idx2trigger[hat]
        # trigger_pred = find_triggers_sentence(trigger_hat)
        # print(trigger_logits.shape, trigger_hat_2d.shape)

        return trigger_hat

    def predict_argument_input(self, words, tokens_x_2d, head_indexes_2d, entity):
        """
        :param tokens_x_2d:
        :param head_indexes_2d:
        :return:
        """
        tokens_x_2d = torch.LongTensor(tokens_x_2d).to(self.device)
        head_indexes_2d = torch.LongTensor(head_indexes_2d).to(self.device)

        self.bert.eval()
        with torch.no_grad():
            encoded_layers = self.bert(tokens_x_2d)
            enc = encoded_layers[0]
        x = enc

        trigger_logits = self.fc_trigger(x)  # 为什么是[1,50,68], 68???
        trigger_hat_2d = trigger_logits.argmax(-1)
        trigger_hat_2d = trigger_hat_2d.squeeze().tolist()
        # hat = max(trigger_hat_2d, key=trigger_hat_2d.count)
        hat = max(trigger_hat_2d)
        trigger_hat = idx2trigger[hat]
        # trigger_pred = find_triggers_sentence(trigger_hat)
        # print(trigger_logits.shape, trigger_hat_2d.shape)

        return trigger_hat

