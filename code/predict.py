import torch.nn as nn
import transformers
from transformers import BertTokenizer

from nerd_tagme import EntityLinkTagMeMatch, get_seed_entities_tagme
import json
import requests
import torch

TRAINING_FILE = "/Users/zhangzihan/Documents/pycharm_project/graduation_project/data/phase1_relevant_fact_selection_trainset.csv"
TOKENIZER = BertTokenizer.from_pretrained("/Users/zhangzihan/Documents/pycharm_project/graduation_project/data/bert-base-uncased-vocab.txt",
                                               do_lower_case = False)
MAX_LEN = 512
MODEL_PATH = "/Users/zhangzihan/Documents/pycharm_project/graduation_project/model/model.bin"



def get_most_entity_id(question):
    tagme_threshold = 0
    TAGME = EntityLinkTagMeMatch(tagme_threshold)
    entity_li = get_seed_entities_tagme(TAGME, question)
    return entity_li


def get_neighborhood(kb_item, p=100, include_labels=True):
    info_url = "https://clocq.mpi-inf.mpg.de/api/neighborhood?item=" + kb_item + "&p=1000"
    response = requests.get(info_url)
    result = response.json()
    return result


def get_qwithc(question, question_id):
    entity_li = get_most_entity_id(question)
    flag = -1
    li = []
    for i in entity_li:
        neighbour_li = get_neighborhood(i[0])
        for sen in neighbour_li:
            flag += 1
            sentence = ""
            for word in sen:
                sentence += word["label"]
                sentence += " and "
            li.append({"question_id": question_id, "qid": flag, "question": question, "context": sentence[:-4]})
    return li


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained("/Users/zhangzihan/Documents/pycharm_project/graduation_project/data/bert-base-uncased")
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        outs = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        bo = self.bert_drop(outs.pooler_output)
        output = self.out(bo)
        return output


def ques_tuple_prediction(ques,tuple,model):
    tokenizer = TOKENIZER
    max_len = MAX_LEN
    # device = torch.device("cuda")
    device = torch.device("cpu")
    inputs = tokenizer.encode_plus(
        ques,
        tuple,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        return_attention_mask=True,
        truncation=True,
        return_tensors='pt')

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    ids = ids.to(device, dtype=torch.long)
    token_type_ids = token_type_ids.to(device, dtype=torch.long)
    mask = mask.to(device, dtype=torch.long)

    outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]


def call_model():
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-d', '--dataset', type=str, default='test')
    args = argparser.parse_args()

    print("Predicting Running")

    device = torch.device("cpu")
    MODEL = BERTBaseUncased()
    model = torch.load(MODEL_PATH)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in model.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    MODEL.load_state_dict(new_state_dict)
    MODEL.to(device)
    MODEL.eval()
    print("load model successfully!")
    return 0


def predict_save(li):
    score_li = []
    MODEL = BERTBaseUncased()
    for i in li:
        question_id = i['question_id']
        qid = i['qid']  # question id starts from 0
        print(qid)
        ques_text = i['question']
        context = i['context']
        score = ques_tuple_prediction(ques_text, context, MODEL)
        score_li.append([question_id, qid, ques_text, context, score])
    sta_score = sorted(score_li, key=(lambda x: x[4]), reverse=True)
    spo_score_file = "/Users/zhangzihan/Documents/pycharm_project/graduation_project/data/predict/spo_score_file.txt"
    f1 = open(spo_score_file, 'a', encoding='utf-8')
    for i in sta_score:
        f1.write(str(i[0]) + ", " + str(i[1]) + ", " + str(i[2]) + ", " + str(i[3]) + ", " + str(i[4]) + "\n")


if __name__ == '__main__':
    # call_model()
    # que_li = []
    # question_set_file = "/Users/zhangzihan/Documents/pycharm_project/graduation_project/data/predict/question_set.txt"
    # f = open(question_set_file, 'r', encoding="utf-8")
    # for line in f.readlines():
    #     line = line.strip("\n")
    #     que_li.append(line.split(", "))
    # for i in que_li:
    #     question = i[1]
    #     question_id = i[0]
    #     qwithc_li = get_qwithc(question, question_id)
    #     print("START SAVING... id is %s \n\n" % question_id)
    #     predict_save(qwithc_li)
    #     print("SAVED! id is %s, question is %s \n\n" % (question_id, question))
    question = "what kind of ford was made first"
    # get context list with question
    li = get_qwithc(question,1)
    for i in li:
        print(i)




