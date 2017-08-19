# Diverse Long Conversation Response Generator
It's a tensorflow implementation of [Generating High-Quality and Informative Conversation Responses with Sequence-to-Sequence Models](https://arxiv.org/abs/1701.03185).
<br/><br/>
Pulled from [Google-Neural-Machine-Translation-GNMT](https://github.com/shawnxu1318/Google-Neural-Machine-Translation-GNMT#google-neural-machine-translation-gnmt).
<br/>

## Prerequisites
* Tensorflow Version: 0.11.0
* Python 3.4+
<br/>

## File Description
### dialog_data_new/
Only listed files for "Questions" here. However, the same is for "Responses" used in training.<br/>Just change "q" in file names to "r".
1. Vocabulary dictionary
    * q_train_dict.pkl is used in training and testing.
2. Original datasets
    * q_train_new
    * q_test_new
    * q_val
Theses files contains sentences that have already been splited. Well we train with Chinese dialogs. No such problem with English.
3. Tokenized datasets
    * q_train_new_tokenized.txt
    * q_test_new_tokenized.txt
    
### check_point/
Stores the parameters in check point.
<br/>
Parameters in it will be used in decoding. Also can be used to continue training.
<br/>
*Notice:
    - Better remove previous ckpt since it takes up a lot of space.
    - Will use previous ckpt in training. So if hyper parameters were changed, than you show probably remove all previous ckpt.
<br/> 

## How to Train
1. Load Data
    Put your training data in dialog_data_new/ and build dictionary. 
<br/>
    data_utils.py have some funcitons about tokenizing, one may help themselves if needed.
<br/>
2. Set Hyperparameters
    In translate.py:
    - Set FLAGS.decode and FLAGS.self-test to False.
    - Change hyper parameters like layer nums, unit nums, etc.
    - Choose the GPU you use.
    - etc
3. Run
    python translate.py
<br/> 

## How to Test
- Set FLAGS.decode to True
- Have to have some ckpt.
