import os
import re
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset
from constant import MIMIC_3_DIR, MIMIC_4_DIR, MIMIC_4_10_DIR
import pandas as pd
import pickle
import warnings
import json
warnings.filterwarnings('ignore', category=FutureWarning)


def proc_text(text):
    text = text.lower().replace("\n", " ").replace("\r", " ")
    text = re.sub('dr\.', 'doctor', text)
    text = re.sub('m\.d\.', 'doctor', text)
    text = re.sub('admission date:', '', text)
    text = re.sub('discharge date:', '', text)
    text = re.sub('--|__|==', '', text)
    return re.sub(r'  +', ' ', text)


def create_main_code(ind2c, version='mimic3'):
    if '10' not in version:
        mc = list(set([c.split('.')[0] for c in set(ind2c.values())]))
        mc.sort()
        ind2mc = {ind: mc for ind, mc in enumerate(mc)}
        mc2ind = {mc: ind for ind, mc in ind2mc.items()}
        return ind2mc, mc2ind
    else:
        try:
            import simple_icd_10 as icd10
            mc = []
            for x in ind2c.values():
                if icd10.is_valid_item(x):
                    mc.append(x[0:3])
            mc = list(set(mc))
            mc.sort()
            ind2mc = {ind: mc for ind, mc in enumerate(mc)}
            mc2ind = {mc: ind for ind, mc in ind2mc.items()}
            return ind2mc, mc2ind
        except:
            mc = list(set([c.split('.')[0] for c in set(ind2c.values())]))
            mc.sort()
            ind2mc = {ind: mc for ind, mc in enumerate(mc)}
            mc2ind = {mc: ind for ind, mc in ind2mc.items()}
            return ind2mc, mc2ind


class MimicFullDataset(Dataset):
    def __init__(self, version, mode, vocab_path, truncate_length,
                 label_truncate_length=30, term_count=1, sort_method='max'):
        self.version = version
        self.mode = mode

        if version in ['mimic3', 'mimic3-50']:
            self.path = os.path.join(MIMIC_3_DIR, f"{version}_{mode}.json")
            self.mimic_version = 'mimic3'
            MIMIC_DIR = MIMIC_3_DIR
        if version in ['mimic4', 'mimic4-50']:
            self.path = os.path.join(MIMIC_4_DIR, f"{version}_{mode}.json")
            self.mimic_version = 'mimic4'
            MIMIC_DIR = MIMIC_4_DIR
        if version in ['mimic4_10', 'mimic4_10-50']:
            self.path = os.path.join(MIMIC_4_10_DIR, f"{version}_{mode}.json")
            self.mimic_version = 'mimic4_10'
            MIMIC_DIR = MIMIC_4_10_DIR

        if version in ['mimic3']:
            self.train_path = os.path.join(MIMIC_3_DIR, "train_full.csv")
        if version in ['mimic3-50']:
            self.train_path = os.path.join(MIMIC_3_DIR, "train_50.csv")

        if version in ['mimic4']:
            self.train_path = os.path.join(MIMIC_4_DIR, "train_full.csv")
        if version in ['mimic4-50']:
            self.train_path = os.path.join(MIMIC_4_DIR, "train_50.csv")

        if version in ['mimic4_10']:
            self.train_path = os.path.join(MIMIC_4_10_DIR, "train_full.csv")
        if version in ['mimic4_10-50']:
            self.train_path = os.path.join(MIMIC_4_10_DIR, "train_50.csv")

        try:
            with open(self.path, "r") as f:
                import ujson
                self.df = ujson.load(f)
                print('Loaded {} records from {}'.format(len(self.df), self.path))
        except:
            with open(self.path, "r") as f:
                import json
                self.df = json.load(f)
                print('Loaded {} records from {}'.format(len(self.df), self.path))

        self.vocab_path = vocab_path
        self.word2id, self.id2word = load_vocab(self.vocab_path)

        self.truncate_length = truncate_length

        self.ind2c, _ = load_full_codes(self.train_path, version=self.mimic_version)
        self.ind2c_full, _ = load_full_codes(os.path.join(MIMIC_DIR, "train_full.csv"), version=self.mimic_version)
        self.c2ind = {c: ind for ind, c in self.ind2c.items()}
        self.code_count = len(self.ind2c)
        if mode == "train":
            print(f'Code count: {self.code_count}')

        self.ind2mc, self.mc2ind = create_main_code(self.ind2c_full, version=self.mimic_version)

        self.main_code_count = len(self.ind2mc)
        if mode == "train":
            print(f'Main code count: {self.main_code_count}')

        from nltk.tokenize import RegexpTokenizer
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.p_codes = {}

        self.len = len(self.df)

        self.label_truncate_length = label_truncate_length
        self.term_count = term_count
        self.sort_method = sort_method
        if self.mode == "train":
            self.prepare_label_feature(self.label_truncate_length)

        try:
            self.icd9sg = pickle.load(open('./mimicdata/ICD9SG.pkl', 'rb'))
        except:
            self.icd9sg = {}

        label_nums = []
        labels = []
        len_words = []
        for case in self.df:
            label_nums.append(len(case['LABELS'].split(';')))
            label = str(case['LABELS']).split(';')
            len_word = case['TEXT'].count(' ') + 1
            len_words.append(len_word)
            label = [x if x[-1] != '.' else x[0:-1] for x in label]
            for l in label:
                if l in self.c2ind:
                    labels.append(self.c2ind[l])
        avg_len_word = np.mean(len_words)
        results = Counter(labels)
        self.rank_index = sorted(results.items(), key=lambda x: x[1], reverse=True)
        self.rank_index = torch.LongTensor([x[0] for x in self.rank_index])
        self.avg_label_num = np.mean(label_nums)

    def __len__(self):
        return self.len

    def gettext(self, index):
        return self.df[index]['TEXT']

    def split(self, text):
        sp = re.sub(r'\n\n+|  +', '\t', text.strip()).replace("\n",
                                                              " ").replace("!", "\t").replace("?", "\t").replace(".",
                                                                                                                 "\t")
        return [s.strip() for s in sp.split("\t") if s.strip()]

    def tokenize(self, text):
        texts = self.split(text)
        all_text = []
        split_text = []
        sentence_index = []
        word_count = 0
        for note in texts:
            now_text = [w.lower() for w in self.tokenizer.tokenize(
                note) if not w.isnumeric()]
            if now_text:
                all_text.extend(now_text)
                split_text.append(now_text)
                word_count += len(now_text)
                sentence_index.append(word_count)
        return all_text, sentence_index, split_text

    def __get_text_label__(self, index):
        text = self.gettext(index)
        label = str(self.df[index]['LABELS']).split(';')
        label = [x if x[-1] != '.' else x[0:-1] for x in label]
        return text, label

    def pad(self, l, pad_token_length, pad_token_id, reverse=False):
        if len(l) > pad_token_length:
            if not reverse:
                return l[0:pad_token_length]
            return l[-pad_token_length:]
        if not reverse:
            return l + [pad_token_id] * (pad_token_length - len(l))
        return [pad_token_id] * (pad_token_length - len(l)) + l

    def text2feature(self, text, truncate_length=None):
        if truncate_length is None:
            truncate_length = self.truncate_length

        all_text, sentence_index, split_text = self.tokenize(text)

        input_word = [self.word2id.get(
            w, self.word2id["**UNK**"]) for w in all_text]
        word_mask = [1] * len(input_word)
        input_word = self.pad(
            input_word, truncate_length, self.word2id["**PAD**"])
        word_mask = self.pad(word_mask, truncate_length, 0)
        word_sent = self.pad(sentence_index, truncate_length, -1)
        return input_word, word_mask, word_sent

    def process(self, text, label):
        input_word, word_mask, word_sent = self.text2feature(text)

        binary_label = [0] * self.code_count
        for l in label:
            if l in self.c2ind:
                binary_label[self.c2ind[l]] = 1

        main_label = [0] * self.main_code_count
        for l in label:
            if l.split('.')[0] in self.mc2ind:
                main_label[self.mc2ind[l.split('.')[0]]] = 1

        return input_word, word_mask, word_sent, \
               binary_label, main_label

    def __getitem__(self, index):
        text, label = self.__get_text_label__(self, index)
        processed = self.process(text, label)
        return processed

    def extract_label_desc(self, ind2c):
        if not hasattr(self, 'desc_dict'):
            if '10' not in self.version:
                self.desc_dict = load_code_descriptions()
            else:
                self.desc_dict = load_code_descriptions('mimic4_10')

        desc_list = []
        for i in ind2c:
            code = ind2c[i]
            code_temp = code
            if not code in self.desc_dict:
                try:
                    from icd9cms import search
                    code_temp = search(code)
                    if code_temp is None or code_temp.alt_code != code:
                        try:
                            code_temp = self.icd9sg[code.replace('.', '')].long_desc
                        except:
                            try:
                                import simple_icd_10 as icd10
                                code_temp = icd10.get_description(code.replace('.', ''))
                            except:
                                code_temp = "Unspecific code {}".format(code)
                    else:
                        code_temp = code_temp.short_desc
                    print(f'Not find desc of {code}')
                    self.p_codes[code] = True
                except:
                    code_temp = "Unspecific code {}".format(code)
            if code_temp is None:
                print(code)
            desc = self.desc_dict.get(code, code_temp)
            desc_list.append(desc)
        return desc_list

    def process_label(self, ind2c, truncate_length, term_count=1, method='max', version='mimic3'):
        desc_list = self.extract_label_desc(ind2c)
        if term_count == 1:
            c_desc_list = desc_list
        else:
            c_desc_list = []
            try:
                with open(f'./embedding/icd_{version}_{method}_sort.json', 'r') as f:
                    import ujson
                    icd_syn = ujson.load(f)
            except:
                try:
                    with open(f'./embedding/icd_{version}_{method}_sort.json', 'r') as f:
                        import json
                        icd_syn = json.load(f)
                except:
                    icd_syn = {}

            for i in ind2c:
                code = ind2c[i]
                tmp_desc = [desc_list[i]]
                new_terms = icd_syn.get(code, [])
                if len(new_terms) >= term_count - 1:
                    tmp_desc.extend(new_terms[0:term_count - 1])
                else:
                    tmp_desc.extend(new_terms)
                    repeat_count = int(term_count / len(tmp_desc)) + 1
                    tmp_desc = (tmp_desc * repeat_count)[0:term_count]
                if i < 5:
                    print(code, tmp_desc)
                c_desc_list.extend(tmp_desc)

        c_input_word = []
        c_word_mask = []
        c_word_sent = []

        for i, desc in enumerate(c_desc_list):
            input_word, word_mask, word_sent = self.text2feature(desc, truncate_length=truncate_length)
            c_input_word.append(input_word)
            c_word_mask.append(word_mask)
            c_word_sent.append(word_sent)

        return c_input_word, c_word_mask, c_word_sent

    def prepare_label_feature(self, truncate_length):
        print('Prepare Label Feature')
        if hasattr(self, 'term_count'):
            term_count = self.term_count
        else:
            term_count = 1
        if hasattr(self, 'sort_method'):
            sort_method = self.sort_method
        else:
            sort_method = 'max'
        c_input_word, c_word_mask, c_word_sent = self.process_label(self.ind2c, truncate_length,
                                                                    term_count=term_count,
                                                                    method=sort_method, version=self.mimic_version)
        mc_input_word, mc_word_mask, mc_word_sent = self.process_label(self.ind2mc, truncate_length, version=self.mimic_version)
        self.c_input_word = torch.LongTensor(c_input_word)
        self.c_word_mask = torch.LongTensor(c_word_mask)
        self.c_word_sent = torch.LongTensor(c_word_sent)
        self.mc_input_word = torch.LongTensor(mc_input_word)
        self.mc_word_mask = torch.LongTensor(mc_word_mask)
        self.mc_word_sent = torch.LongTensor(mc_word_sent)


def my_collate_fn(batch):
    type_count = len(batch[0])
    batch_size = len(batch)
    output = ()
    max_length = 0
    for sub in batch:
        max_length = max(max_length, np.sum(sub[1]))
    for i in range(type_count):
        tmp = []
        for item in batch:
            if i <= 2:
                tmp.extend(item[i][0:max_length])
            else:
                tmp.extend(item[i])
        if len(tmp) <= batch_size:
            output += (torch.LongTensor(tmp),)
        elif isinstance(tmp[0], int):
            output += (torch.LongTensor(tmp).reshape(batch_size, -1),)
        elif isinstance(tmp[0], float):
            output += (torch.FloatTensor(tmp).reshape(batch_size, -1),)
        elif isinstance(tmp[0], list):
            dim_y = len(tmp[0])
            if isinstance(tmp[0][0], int):
                output += (torch.LongTensor(tmp).reshape(batch_size, -1, dim_y),)
            elif isinstance(tmp[0][0], float):
                output += (torch.FloatTensor(tmp).reshape(batch_size, -1, dim_y),)
    return output


def load_vocab(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        words = [line.strip().split()[0] for line in lines]
    except BaseException:
        if path.endswith('.model'):
            import gensim.models
            model = gensim.models.Word2Vec.load(path)
        if path.endswith('.bin'):
            import gensim.models
            model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
        words = list(model.wv.vocab)
        del model

    try:
        with open('./embedding/word_count_dict.json', 'r') as f:
            import ujson
            word_count_dict = ujson.load(f)
    except BaseException:
        try:
            with open('./embedding/word_count_dict.json', 'r') as f:
                import json
                word_count_dict = json.load(f)
        except:
            with open('../embedding/word_count_dict.json', 'r') as f:
                import json
                word_count_dict = json.load(f)

    words = [w for w in words if w in word_count_dict]

    for w in ["**UNK**", "**PAD**", "**MASK**"]:
        if not w in words:
            words = words + [w]
    word2id = {word: idx for idx, word in enumerate(words)}
    id2word = {idx: word for idx, word in enumerate(words)}
    return word2id, id2word


def load_full_codes(train_path, version='mimic3'):
    from collections import defaultdict
    desc_dict = load_code_descriptions(version=version)
    if version == 'mimic2':
        pass
    elif version == 'mimic3' or version == 'mimic4':
        codes = set()
        for split in ['train', 'dev', 'test']:
            with open(train_path.replace('train', split), 'r') as f:
                import csv
                lr = csv.reader(f)
                next(lr)
                for row in lr:
                    for code in row[3].split(';'):
                        if len(code) == 0:
                            continue
                        if code[-1] == '.':
                            code = code[0:-1]
                        codes.add(code)
        codes = set([c for c in codes if c != ''])
        ind2c = defaultdict(str, {i: c for i, c in enumerate(sorted(codes))})
    else:
        codes = set()
        for split in ['train', 'dev', 'test']:
            with open(train_path.replace('train', split), 'r') as f:
                import csv
                lr = csv.reader(f)
                next(lr)
                for row in lr:
                    for code in row[3].split(';'):
                        if len(code) == 0:
                            continue
                        if code[-1] == '.':
                            code = code[0:-1]
                        codes.add(code)
        ind2c = defaultdict(str, {i: c for i, c in enumerate(sorted(codes))})
    return ind2c, desc_dict


def reformat(code, is_diag):
    code = ''.join(code.split('.'))
    if is_diag:
        if code.startswith('E'):
            if len(code) > 4:
                code = code[:4] + '.' + code[4:]
        else:
            if len(code) > 3:
                code = code[:3] + '.' + code[3:]
    else:
        code = code[:2] + '.' + code[2:]
    return code


def reformat10(code, is_diag):
    code = ''.join(code.split('.'))
    if is_diag:
        code = code[:3] + '.' + code[3:]
    else:
        code = code
    return code


def load_code_descriptions(version='mimic3'):
    from collections import defaultdict
    desc_dict = defaultdict(str)
    DATA_DIR = "./mimicdata/"

    if version == 'mimic2':
        pass
    elif version == 'mimic3' or version == 'mimic4':
        desc_dict['36.01'] = 'Single vessel percutaneous transluminal coronary angioplasty [PTCA] or coronary atherectomy without mention of thrombolytic agent'
        desc_dict['36.02'] = 'Single vessel percutaneous transluminal coronary angioplasty [PTCA] or coronary atherectomy with mention of thrombolytic agent'
        desc_dict['36.05'] = 'Multiple vessel percutaneous transluminal coronary angioplasty [PTCA] or coronary atherectomy performed during the same operation, with or without mention of thrombolytic agent'
        desc_dict['488.1'] = 'Influenza due to identified novel H1N1 influenza virus'
        desc_dict['719.70'] = 'Difficulty in walking involving joint site unspecified'
        try:
            with open("%s/D_ICD_DIAGNOSES.csv" % (DATA_DIR), 'r') as descfile:
                import csv
                r = csv.reader(descfile)
                next(r)
                for row in r:
                    code = row[1]
                    desc = row[-1]
                    desc_dict[reformat(code, True)] = desc
        except:
            pass
        try:
            with open("%s/D_ICD_PROCEDURES.csv" % (DATA_DIR), 'r') as descfile:
                import csv
                r = csv.reader(descfile)
                next(r)
                for row in r:
                    code = row[1]
                    desc = row[-1]
                    if code not in desc_dict.keys():
                        desc_dict[reformat(code, False)] = desc
        except:
            pass
        try:
            with open('%s/ICD9_descriptions' % DATA_DIR, 'r') as labelfile:
                for _, row in enumerate(labelfile):
                    row = row.rstrip().split()
                    code = row[0]
                    if code not in desc_dict.keys():
                        desc_dict[code] = ' '.join(row[1:])
        except:
            pass
    else:
        try:
            with open("%s/d_icd10_diagnoses.csv" % (DATA_DIR), 'r') as descfile:
                import csv
                r = csv.reader(descfile)
                next(r)
                for row in r:
                    if row[1] == '10':
                        code = row[0]
                        desc = row[-1]
                        desc_dict[code] = desc
        except:
            pass
        try:
            with open("%s/d_icd10_procedures.csv" % (DATA_DIR), 'r') as descfile:
                import csv
                r = csv.reader(descfile)
                next(r)
                for row in r:
                    if row[1] == '10':
                        code = row[0]
                        desc = row[-1]
                        if code not in desc_dict.keys():
                            desc_dict[code] = desc
        except:
            pass

    return desc_dict