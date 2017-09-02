import os
import pickle

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

q_dict_name = "q_train_dict.pkl"
r_dict_name = "r_train_dict.pkl" 

def tokenize_sentence(dict, sentence):
    """
    Tokenize the given sentence with reference to dict.

    Args: dict: A vocabulary dictionary. (key:word, val:id).
          sentence: A sentence that have been split.

    Returns: rst: a list representing the tokenized sentence.
    """
    rst = []
    for word in sentence.split():
        if word.encode('utf8') not in dict.keys():
            rst.append(UNK_ID)
        else:
            rst.append(dict[word.encode('utf8')])
    return rst


def list_to_str(lis):
    """
    Turn a list to str. Each element is seperated by a space.

    Args: lis: A list.

    Returns: A str that concatenates each elements in lis by a space.
        Ends with '\n'.
    """
    rst = ""
    for i in lis:
        rst += (str(i) + " ")
    rst += '\n'
    return rst

def tokenize_dataset(dict_, file_):
    """
    Tokenize the whole dataset and write it to file.

    Args: dict_: A dictionary that contains the words.
          file_: The path of the dataset.
    
    Returns: output_filename: The path of the tokenized dataset.
    """
    print("==============Tokenizing %s================" % file_)
    fin = open(file_, "r")
    output_filename = file_ + "_tokenized.txt"
    if os.path.exists(output_filename):
        return output_filename
    fout = open(output_filename, "w")
    cnt = 0
    for sentence in fin.readlines():
        cnt += 1
        if cnt % 10000 == 0:
            print("Processed %d sentences." % cnt)
        fout.write(list_to_str(tokenize_sentence(dict_, sentence)))
    fout.close()
    fin.close()
    return output_filename


def get_vocab(data_dir):
    """
    Get the vocabulary dictionary that contains Q & R words.

    Args: data_dir: Where the dictionary stores.

    Returns: Two dictionaries that contains Q & R words respectively.
        (key:word, val:id)
    """
    # q_dict contains all the words in q_train... I guess.
    q_dict_filename = os.path.join(data_dir, q_dict_name)
    r_dict_filename = os.path.join(data_dir, r_dict_name)
    q_dict = pickle.load(open(q_dict_filename, 'rb'), encoding='iso-8859-1')
    r_dict = pickle.load(open(r_dict_filename, 'rb'), encoding='iso-8859-1')
    return q_dict, r_dict


def get_reverse_vocab_dict(dict_ori):
    """
    Switch the key and value of the original dictionary.

    Args: dict_ori: A dictionary in which elements are (key:word, value:id).

    Returns: dict_r: A dictionaory in which elements are (key:id, value:word).
    """
    dict_r = {}
    print(type(dict_ori))
    for key, val in dict_ori.items():
        dict_r[val] = key
    return dict_r


def build_vocab_dict(data_dir, train_data, data_type, vocab_size):
    """
    Build vocabulary dictionary for Q / R training file.
    The vocab dict will be an ordered dictionary with
    (key:word, val:id), and then wrote to pkl file.

    Args: data_dir: Where to store the dictionary
        train_data: The path of the training data.
        data_type: "q" / "r"
    """
    print("==============Building vocabulary for %s================" % train_data)
    if data_type == q_dict_name[:1]:
        dict_filename = os.path.join(data_dir, q_dict_name)
    else:
        dict_filename = os.path.join(data_dir, r_dict_name)
    if os.path.exists(dict_filename):
        return
    vocab = {}  # key:word, val:id
    vocab[_PAD] = PAD_ID  # 0
    vocab[_GO] = GO_ID  # 1
    vocab[_EOS] = EOS_ID  # 2
    vocab[_UNK] = UNK_ID  # 3
    count = {}  # key:word, val:how many times this word appears
    fin = open(train_data, "r")
    for sentence in fin.readlines():
        for word in sentence.split():
            if word.encode('utf8') not in vocab.keys():
                count[word.encode('utf8')] = 1
            else:
                count[word.encode('utf8')] += 1
    fin.close()
    sorted_count = sorted(count.items(), key=lambda tup: tup[1], reverse=True)
    word_id = 0
    while word_id < vocab_size - len(_START_VOCAB):
        vocab[sorted_count[word_id][0]] = word_id + len(_START_VOCAB)
        word_id += 1
    fw = open(dict_filename, 'wb')
    pickle.dump(vocab, fw, pickle.HIGHEST_PROTOCOL)
    fw.close()
    return


def prepare_data(data_dir, q_vocab_size, r_vocab_size):
    """Create vocabularies for Q & R. Tokenize data.

    Args:
        data_dir: directory in which the data sets will be stored.

    Returns:
        A tuple of 4 elements:
            (1) path to the token-ids for English training data-set,
            (2) path to the token-ids for French training data-set,
            (3) path to the token-ids for English development data-set,
            (4) path to the token-ids for French development data-set,
    """
    # q_train_new contains sentences that have been split
    q_train_raw_filename = os.path.join(data_dir, "q_train_new")
    r_train_raw_filename = os.path.join(data_dir, "r_train_new")
    q_test_raw_filename = os.path.join(data_dir, "q_test_new")
    r_test_raw_filename = os.path.join(data_dir, "r_test_new")

    # build dictionary
    build_vocab_dict(data_dir, q_train_raw_filename, "q", q_vocab_size)
    build_vocab_dict(data_dir, r_train_raw_filename, "r", r_vocab_size)

    # get vocabulary lists from q and r.
    q_dict, r_dict = get_vocab(data_dir)

    # tokenize dataset
    q_train_filename = tokenize_dataset(q_dict, q_train_raw_filename)
    q_test_filename = tokenize_dataset(q_dict, q_test_raw_filename)
    r_train_filename = tokenize_dataset(r_dict, r_train_raw_filename)
    r_test_filename = tokenize_dataset(r_dict, r_test_raw_filename)

    return q_train_filename, r_train_filename,\
           q_test_filename, r_test_filename

prepare_data("dialog_data_new")
