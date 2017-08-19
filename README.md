About this file:
Contains the code and data for Google's paper-"Generating High-Quality Response with Seq2Seq".

About dialog_data_new/:
(1)Vocabulary dictionary:
    q_train_dict.pkl is used in training and testing.
    q_train_dict_old.pkl is the original version, does not contain the default IDs like GO, PAD, UNK, etc. Have some unknown problem with encoding, so it leads to a lot of UNK after tokenizing dataset.
(2)Original datasets:
    q_train_new
    q_test_new
    q_val
    These files contains sentences that have been split.
(3)Tokenized datasets:
    q_train_new_tokenized.txt
    q_test_new_tokenized.txt
    
About check_point/:
Stores the parameters in check point.
Parameters in it will be used in decoding.
*Notice:
- Better remove previous ckpt since it takes up a lot of space.
- Will use previous ckpt in training. So if hyper parameters were changed, than you show probably remove all previous ckpt.

About training:
- Set FLAGS.decode/self-test to False.
- Change hyper parameters like layer nums, unit nums, etc.

About testing:
- Set FLAGS.decode to True
