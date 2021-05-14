"""
This example demonstrates how to use `CNN` model from
`speechemotionrecognition` package
"""
from keras.utils import np_utils
from utils.dnn import CNN, LSTM, CNN_InterSpeech_2017, Manon2020
from utils.utilities import get_feature_vector_from_mfcc
import numpy as np
import os
from sklearn.model_selection import train_test_split
from utils.utilities import get_data, get_feature_vector_from_mfcc, get_data_IEMCAP
import argparse

_DATA_PATH = '.dataset'
_CLASS_LABELS = ("Neutral", "Angry", "Happy", "Sad")
to_flatten = False

def extract_data(flatten, _DATA_PATH=_DATA_PATH):
    data, labels = get_data_IEMCAP(_DATA_PATH, class_labels=_CLASS_LABELS,
                            flatten=flatten)
    x_train, x_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=0.2,
        random_state=42)

    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(
        y_test), len(_CLASS_LABELS)


def data_model_define(EMOCAP_path, type):

    # --------------------------------------------- Data processing --------------------------------------------- #
    # Download http://sail.usc.edu/iemocap/ in IEMOCAP folder, unzipping IEMOCAP_full_release_withoutVideos.tar
    data_path = EMOCAP_path
    x_train, x_test, y_train, y_test, num_labels = extract_data(
        flatten=to_flatten, _DATA_PATH=data_path)
    y_train = np_utils.to_categorical(y_train)
    y_test_train = np_utils.to_categorical(y_test)
    in_shape = x_train[0].shape
    # --------------------------------------------- Model selection --------------------------------------------- #
    if type == 'cnn':
        x_train = x_train.reshape(x_train.shape[0], in_shape[0], in_shape[1], 1)
        tmp_x_test = x_test
        x_test = x_test.reshape(x_test.shape[0], in_shape[0], in_shape[1], 1)
        model = CNN(input_shape=x_train[0].shape,
                    num_classes=num_labels)
    elif type == 'rnn':
        tmp_x_test = x_test
        model = LSTM(input_shape=x_train[0].shape,
                     num_classes=num_labels)
    elif type == 'inter2017':
        x_train = x_train.reshape(x_train.shape[0], in_shape[0], in_shape[1], 1)
        tmp_x_test = x_test
        x_test = x_test.reshape(x_test.shape[0], in_shape[0], in_shape[1], 1)
        model = CNN_InterSpeech_2017(input_shape=x_train[0].shape,
                                     num_classes=num_labels)
    elif type == 'Manon2020':
        tmp_x_test = x_test
        model = Manon2020(input_shape=x_train[0].shape,
                          num_classes=num_labels)
    else:
        raise Exception('Choose correct model name')

    return model, x_train, y_train, x_test, y_test, y_test_train, tmp_x_test


def EarningCall_test(model, EarningCall_path):
    # inference in list
    # Download https://github.com/GeminiLn/EarningsCall_Dataset in Amazon, 3M, Twitter an so on.. wav files in directory
    # Amazon, 3M, Twitter =
    # './EarningsCall_Dataset/Amazon_Inc_20170202/Wav', '/3M_Company_20170425/Wav/', '/Twitter_Inc_20170209/Wav'
    filename_root = EarningCall_path

    list_wav = []
    for (path, dir, files) in os.walk(filename_root):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.wav':
                # print("%s/%s" % (path, filename))
                list_wav.append(os.path.join(path, filename))

    for i, mp3_file in enumerate(list_wav):
        print('file name : ', os.path.split(mp3_file)[1])
        pred_cls = model.predict_one(get_feature_vector_from_mfcc(mp3_file, flatten=to_flatten), type)
        if pred_cls == 0:
            print('prediction : Neutral\n')
        elif pred_cls == 1:
            print('prediction : Angry\n')
        elif pred_cls == 2:
            print('prediction : Happy\n')
        elif pred_cls == 3:
            print('prediction : Sad\n')


def EMOCAP_test(model, EMOCAP_path, type):
    _, _, _, x_test, y_test, _, tmp_x_test = data_model_define(EMOCAP_path, type)
    if type == 'inter2017' or 'cnn':
        x_test = tmp_x_test
    model.evaluate(x_test, y_test, type)

def inference(type, select_test_data, EMOCAP_path, EarningCall_path, saved_model_root):
    shapepe_define = np.zeros((198, 39))
    input_shape_define = np.zeros((271, 198, 39, 1))

    if saved_model_root:
        model_name = saved_model_root
    else:
        # cnn default
        if type == 'cnn':
            model_name = './models/model_EMOCAP_CNN_51.3.h5'
        elif type == 'rnn':
            model_name = './models/model_EMOCAP_LSTM_60.7.h5'
        elif type == 'inter2017':
            model_name = './models/model_EMOCAP_Inter2017_64.3.h5'
        elif type == 'Manon2020':
            model_name = './models/model_EMOCAP_Manon2020_64.5.h5'
        else:
            raise NameError('select correct name')
    #
    if type == 'cnn':
        model = CNN(input_shape=shapepe_define.shape,
                    num_classes=4)
        model.load_model(model_name)
    elif type == 'rnn':
        model = LSTM(input_shape=shapepe_define.shape,
                     num_classes=4)
        model.load_model(model_name)

    elif type == 'inter2017':
        model = CNN_InterSpeech_2017(input_shape=shapepe_define.shape,
                                     num_classes=4)
        model.load_model(model_name)

    elif type == 'Manon2020':
        model = Manon2020(input_shape=shapepe_define.shape,
                     num_classes=4)
        model.load_model(model_name)

    print("Neutral", "Angry", "Happy", "Sad")

    if select_test_data == 'EarningCall':
        # EarningCall test
        EarningCall_test(model, EarningCall_path)
    elif select_test_data == 'EMOCAP':
        # EMOCAP test
        EMOCAP_test(model, EMOCAP_path, type)


def train(type, saved_model_path, epoch, EMOCAP_path):
    # --------------------------------------------- Data & Model Define --------------------------------------------- #
    # For EMOCAP database,
    model, x_train, y_train, x_test, y_test, y_test_train, tmp_x_test = data_model_define(EMOCAP_path, type)

    # --------------------------------------------- Train & Evaluation --------------------------------------------- #
    model.train(x_train, y_train, x_test, y_test_train, epoch, saved_model_path)
    print('Training Done')
    # model.save_model_path(saved_model_path)

    # Evaluation
    if type == 'inter2017' or 'cnn':
        x_test = tmp_x_test
    model.evaluate(x_test, y_test, type)

    print('Done :)')

    # model path return
    return saved_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Speech emotion recognition')

    model_name_default = './models/best_model_CNN.h5'
    model_type_default = 'cnn'
    saved_model_root_default = False
    epoch_defult = 50
    select_test_data_default = 'EMOCAP'
    EMOCAP_path_default = '../dataset/IEMOCAP/IEMOCAP_full_release'
    EarningCall_path_default = './EarningsCall_Dataset/Amazon_Inc_20170202/Wav'
    load_model_default = False

    parser.add_argument('--model_type', type=str, help='selection model type',
                        choices=['cnn', 'rnn', 'inter2017', 'Manon2020'],
                        default=model_type_default)
    parser.add_argument('--save_model', type = str, help = 'save model name for .h5',
                        default = model_name_default)
    parser.add_argument('--is_train', action='store_true',
                        help="training")
    parser.add_argument('--epoch', type=int, help='epoch number',
                        default=epoch_defult)
    parser.add_argument('--select_test_data', type=str, help='selection test dataset',
                        choices=['EMOCAP', 'EarningCall'],
                        default=select_test_data_default)
    parser.add_argument('--EMOCAP_path', type=str, help='load EMOCAP data',
                        default=EMOCAP_path_default)
    parser.add_argument('--EarningCall_path', type=str, help='load EarningCall data',
                        default=EarningCall_path_default)
    parser.add_argument('--load_model', type=str, help='load inference model',
                        default=load_model_default)

    argv = parser.parse_args()

    model_type = argv.model_type
    model_name = argv.save_model
    is_train = argv.is_train
    epoch = argv.epoch
    select_test_data = argv.select_test_data
    EMOCAP_path = argv.EMOCAP_path
    EarningCall_path = argv.EarningCall_path
    load_model = argv.load_model

    # Training and Test
    if is_train:
        saved_model_name = train(model_type, model_name, epoch, EMOCAP_path)
        inference(model_type, select_test_data, EMOCAP_path, EarningCall_path, model_name)
    else:
        # # only Test
        inference(model_type, select_test_data, EMOCAP_path, EarningCall_path, load_model)