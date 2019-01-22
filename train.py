from argparse import ArgumentParser
import time
import json
import numpy as np
import csv
import subprocess
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as D
from torch.autograd import Variable
from sklearn.metrics import f1_score

import data_pro as pro
from pyt_CR_CNN import CR_CNN, PairwiseRankingLoss


def create_csv(fpath):
    with open(fpath, 'w') as f:
        csv.writer(f).writerow(['timestamp', 'test_f1', 'optimizer', 'batch_size', 'epochs', 'lr', 'decay', 'n_filters', 'p_dropout', 'max_len', 'pos_embed_size', 'n_pos_embed', 'window', 'n_class'])
    f.close()
# end def


def main():
    print('\n---------------------------------------------- Setup -----------------------------------------------')

    parser = ArgumentParser(description='')
    parser.add_argument('--max_len', type=int, metavar='<MAX_LEN>', default=123, help='max_len')
    parser.add_argument('--pos_embed_size', type=int, metavar='<POS_EMBED_SIZE>', default=70, help='position_embedding_size')
    parser.add_argument('--n_pos_embed', type=int, metavar='<N_POS_EMBED>', default=123, help='position_embedding_num')
    parser.add_argument('--window', type=int, metavar='<WINDOW>', default=3, help='slide_window')
    parser.add_argument('--n_filters', type=int, metavar='<n_filters>', default=1000, help='num_filters')
    parser.add_argument('--p_dropout', type=float, metavar='<p_dropout>', default=0.5, help='keep_prob')
    parser.add_argument('--epochs', type=int, metavar='<EPOCHS>', default=50, help='number of epochs')
    parser.add_argument('--lr', type=float, metavar='<LR>', default=0.001, help='learning_rate')
    parser.add_argument('--decay', type=float, metavar='<decay>', default=0, help='weight_decay')
    parser.add_argument('--batch_size', type=int, metavar='<BATCH_SIZE>', default=32, help='batch_size')
    parser.add_argument('--opt', type=str, metavar='<OPT>', default='adam', help='optimizer: adam or sgd')
    A = parser.parse_args()

    N_CLASS = 19        # class_num
    N_EPOCHS = A.epochs
    MAX_LEN = A.max_len        # max_len
    POS_EMBED_SIZE = A.pos_embed_size        # position_embedding_size
    N_POS_EMBED = A.n_pos_embed       # position_embedding_num
    WINDOW = A.window          # slide_window
    BATCH_SIZE = A.batch_size
    n_filters = A.n_filters      # num_filters
    p_dropout = A.p_dropout       # keep_prob
    LR = A.lr     # learning_rate
    DECAY = A.decay     # learning rate decay
    OPT = A.opt
    TIMESTAMP = time.strftime("%Y%m%d-%H%M")
    FPATH_BEST_MODEL = 'saved_models/20190122/crcnn_opt-{}_epoch-{}_lr-{}_decay-{}_{}.pkl'.format(OPT, N_EPOCHS, LR, DECAY, TIMESTAMP)

    print('Parameters:\n{}'.format(dict(MAX_LEN=MAX_LEN,
                                        POS_EMBED_SIZE=POS_EMBED_SIZE,
                                        N_POS_EMBED=N_POS_EMBED,
                                        N_CLASS=N_CLASS,
                                        n_filters=n_filters,
                                        p_dropout=p_dropout,
                                        WINDOW=WINDOW,
                                        LR=LR,
                                        DECAY=DECAY,
                                        BATCH_SIZE=BATCH_SIZE,
                                        EPOCHS=N_EPOCHS,
                                        OPT=OPT,
                                        TIMESTAMP=TIMESTAMP
                                        )))

    # print('\n---------------------------------------------- Load Data -----------------------------------------------')

    data_train_valid = pro.load_data('data/nine_train.txt')

    concat = list(zip(data_train_valid[0], data_train_valid[1], data_train_valid[2], data_train_valid[3]))
    data_train, data_validation = train_test_split(concat, test_size=0.2, random_state=0)
    # print(data_train[0])

    new_data_train = [i for i in zip(*data_train)]
    new_data_validation = [i for i in zip(*data_validation)]

    word_dict = pro.build_dict(new_data_train[0] + new_data_validation[0])      # word_dict: 19215 words and their id
    print('len(word_dict): ', len(word_dict))

    sent_train, y_train, dist1_train, dist2_train = pro.vectorize(new_data_train, word_dict, MAX_LEN)
    y_train = np.array(y_train).astype(np.int64)
    X_train = np.concatenate((sent_train, np.array(dist1_train), np.array(dist2_train)), 1)
    print('Data shape: X_train={}, y_train={}'.format(X_train.shape, y_train.shape))

    sent_valid, y_valid, dist1_valid, dist2_valid = pro.vectorize(new_data_validation, word_dict, MAX_LEN)
    y_valid = np.array(y_valid).astype(np.int64)
    X_valid = np.concatenate((sent_valid, np.array(dist1_valid), np.array(dist2_valid)), 1)
    print('Data shape: X_valid={}, y_valid={}'.format(X_valid.shape, y_valid.shape))

    # fpath_embedding = '../relation-extraction-ly-dev/data/pre_trained_embeddings/glove.6B.300d.txt'
    # embedding_matrix = pro.load_glove_embeddings(fpath_embedding, word_dict)
    # print('Pre-trained embeddings loaded from <{}>.'.format(fpath_embedding))
    # np.save('data/embedding_matrix.npy', embedding_matrix)
    embedding_matrix = np.load('data/embedding_matrix.npy')

    print('\n---------------------------------------------- Build Model -----------------------------------------------')

    model = CR_CNN(MAX_LEN, embedding_matrix, POS_EMBED_SIZE, N_POS_EMBED, WINDOW, N_CLASS, n_filters, p_dropout).cuda()
    print(model)

    loss_func = PairwiseRankingLoss(N_CLASS)
    if OPT == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=DECAY)
    elif OPT == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=DECAY)
    # end if

    print('\n------------------------------------------------- Train --------------------------------------------------')

    def data_unpack(cat_data, target):
        list_x = np.split(cat_data.numpy(), [MAX_LEN, MAX_LEN + N_POS_EMBED], 1)
        batch_x = Variable(torch.from_numpy(list_x[0])).cuda()
        batch_d1 = Variable(torch.from_numpy(list_x[1])).cuda()
        batch_d2 = Variable(torch.from_numpy(list_x[2])).cuda()
        target = Variable(target).cuda()
        return batch_x, batch_d1, batch_d2, target

    def prediction(sc, y):
        '''
        Calculat the f1 score for y_true and y_predict.

        c_target_dict: 19 relation label -> 19 relation name
        tr_target_dict: 19 relation name -> 10 relation label

        '''
        y_true = y.cpu().data.numpy()
        y_predict = torch.max(sc, 1)[1].long().cpu().data.numpy()
        f1 = f1_score(y_true, y_predict, average='micro')
        return f1 * 100
    # end def

    best_score = 0
    patience = 0
    for i in range(1, N_EPOCHS + 1):
        patience += 1

        # train over batches
        tensor_x_train = torch.from_numpy(X_train.astype(np.int64))
        tensor_y_train = torch.LongTensor(y_train)
        train_datasets = D.TensorDataset(data_tensor=tensor_x_train,
                                         target_tensor=tensor_y_train)
        train_dataloader = D.DataLoader(dataset=train_datasets,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,
                                        num_workers=2)
        score_train = 0
        loss = 0
        n_trained_batch = 0
        for (batch_x_cat, batch_y) in train_dataloader:
            n_trained_batch += 1
            batch_x, batch_d1, batch_d2, batch_y = data_unpack(batch_x_cat, batch_y)
            # print('batch_x: ', batch_x.shape)
            # print('batch_d1: ', batch_d1.shape)
            # print('batch_d2: ', batch_d2.shape)
            weight_o = model(batch_x, batch_d1, batch_d2)
            loss_per_batch = loss_func(weight_o, batch_y)
            optimizer.zero_grad()
            loss_per_batch.backward()
            optimizer.step()
            loss += loss_per_batch
            score_train += prediction(weight_o, batch_y)
        # end for
        loss = loss.cpu().data.numpy()[0] / n_trained_batch
        score_train = score_train / n_trained_batch

        # evaluate over batches
        tensor_X_valid = torch.from_numpy(X_valid.astype(np.int64))
        tensor_y_valid = torch.LongTensor(y_valid)
        valid_datasets = D.TensorDataset(data_tensor=tensor_X_valid,
                                         target_tensor=tensor_y_valid)
        valid_dataloader = D.DataLoader(dataset=valid_datasets,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,
                                        num_workers=2)
        score_val = 0
        n_eval_batch = 0
        for (batch_x_cat, batch_y) in valid_dataloader:
            batch_x, batch_d1, batch_d2, batch_y = data_unpack(batch_x_cat, batch_y)
            weight_o = model(batch_x, batch_d1, batch_d2, False)
            score_val += prediction(weight_o, batch_y)
            n_eval_batch += 1
        # end for
        score_val = score_val / n_eval_batch
        # if i % 10 == 0:
        print('Epoch [{}/{}]\t train_loss: {:4f}\t train_f1: {:.3f}\t test_f1: {:.3f}'.format(i, N_EPOCHS, loss, score_train, score_val))

        # save best model
        current_score = score_val
        if current_score > best_score:
            patience = 0
            best_score = current_score
            torch.save(model.state_dict(), FPATH_BEST_MODEL)
            print('Model saved to <{}>'.format(FPATH_BEST_MODEL))

        if patience >= 10:
            print('Earlystopping: patience = {}'.format(patience))
            break
        # end for

    with open('saved_models/20190122/results_20190122.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([TIMESTAMP, round(best_score, 3), OPT, BATCH_SIZE, N_EPOCHS, LR, DECAY, n_filters, p_dropout, MAX_LEN, POS_EMBED_SIZE, N_POS_EMBED, WINDOW, N_CLASS])
    f.close()

    print('\n------------------------------------------------- Test --------------------------------------------------\n')

    # model = torch.load('saved_models/20190122/crcnn_opt-adam_epoch-50_lr-0.001_decay-0_20190122-2049.pkl')

    # test
    data_test = pro.load_data('data/nine_test.txt')
    sent_test, y_test, dist1_test, dist2_test = pro.vectorize(data_test, word_dict, MAX_LEN)
    y_test = np.array(y_test).astype(np.int64)
    X_test = np.concatenate((sent_test, np.array(dist1_test), np.array(dist2_test)), 1)
    print('Data shape: X_test={}, y_test={}'.format(X_test.shape, y_test.shape))

    # evaluate on test set
    tensor_X_test = torch.from_numpy(X_test.astype(np.int64))
    tensor_y_test = torch.LongTensor(y_test)
    test_datasets = D.TensorDataset(data_tensor=tensor_X_test,
                                    target_tensor=tensor_y_test)
    test_dataloader = D.DataLoader(dataset=test_datasets,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=2)
    score_test = 0
    n_test_batch = 0
    y_predict_test = []
    y_true_test = []
    for (batch_x_cat, batch_y) in test_dataloader:
        batch_x, batch_d1, batch_d2, batch_y = data_unpack(batch_x_cat, batch_y)
        weight_o = model(batch_x, batch_d1, batch_d2, False)

        y_true_test.extend(list(batch_y.cpu().data.numpy()))
        y_predict_test.extend(list(torch.max(weight_o, 1)[1].long().cpu().data.numpy()))

        score_test += prediction(weight_o, batch_y)
        n_test_batch += 1
    # end for
    score_test = score_test / n_test_batch
    print('score_test={:.3f}'.format(score_test))

    # save y_predict to txt file and run official scorer
    target_dict = json.load(open('data/target_dict.txt', 'r', encoding='utf-8'))  # 19 classes
    c_target_dict = {value: key for key, value in target_dict.items()}      # label -> name

    y_predict_test_names = [c_target_dict[i] for i in y_predict_test]
    y_true_test_names = [c_target_dict[i] for i in y_true_test]

    FPATH_Y_PRED_TXT = 'saved_models/20190122/y_predict_{}.txt'.format(TIMESTAMP)
    FPATH_Y_TRUE_TXT = 'saved_models/20190122/y_true_{}.txt'.format(TIMESTAMP)
    with open(FPATH_Y_PRED_TXT, 'w') as f:
        for i, p in enumerate(y_predict_test_names):
            f.write('{}\t{}'.format(i, p))
            f.write('\n')
    f.close()

    with open(FPATH_Y_TRUE_TXT, 'w') as f:
        for i, t in enumerate(y_true_test_names):
            f.write('{}\t{}'.format(i, t))
            f.write('\n')
    f.close()

    print('TXT files saved to <{}> and <{}>'.format(FPATH_Y_PRED_TXT, FPATH_Y_TRUE_TXT))

    PERL_PATH = 'data/semeval2010_task8_scorer-v1.2.pl'
    process = subprocess.Popen(["perl", PERL_PATH, FPATH_Y_PRED_TXT, FPATH_Y_TRUE_TXT], stdout=subprocess.PIPE)
    for line in str(process.communicate()[0].decode("utf-8")).split("\\n"):
        print(line)

    print('\n------------------------------------------------- END --------------------------------------------------\n\n\n')

# end def


if __name__ == '__main__':
    # create_csv('results.csv')
    main()
