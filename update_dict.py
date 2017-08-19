import data_utils
import pickle

"""
现在的问题是，数据集中规定好的那些符号和原来GNMT里面的不一样。所以需要更新一下咯。
"""

def update_dict(dict_ori, src_of_vocab):
    """
    Args: dict_ori is a dict that needs to be updated.
        src_of_vocab: str, "q" or "r"

    Returns: returns an updated dictionary that adds the special tags.
    """
    dict_new = {}
    dict_new[data_utils._PAD] = data_utils.PAD_ID  # 0
    dict_new[data_utils._GO] = data_utils.GO_ID  # 1
    dict_new[data_utils._EOS] = data_utils.EOS_ID  # 2
    dict_new[data_utils._UNK] = data_utils.UNK_ID  # 3
    for (k, v) in dict_ori.items():
        dict_new[k] = v + 3  # originaly, value starts from 1
    fw = open(src_of_vocab + "_train_vocab.pkl", 'wb')
    pickle.dump(dict_new, fw, pickle.HIGHEST_PROTOCOL)
    fw.close()
    print("the %s dict contains %d words" % (src_of_vocab, len(dict_new.keys())))

q_dict, r_dict = data_utils.get_vocab("dialog_data_new")
update_dict(q_dict, "q")
update_dict(r_dict, "r")
