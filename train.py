from argparse import ArgumentParser
import time
import json
import numpy as np
import csv

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
    fpath_best_model = 'saved_models/20190118/crcnn_opt_{}_epoch-{}_lr-{}_decay-{}_{}.pkl'.format(OPT, N_EPOCHS, LR, DECAY, TIMESTAMP)

    # OPT = 'adam'
    # MAX_LEN = 123        # max_len
    # POS_EMBED_SIZE = 70        # position_embedding_size
    # N_POS_EMBED = 123       # position_embedding_num
    # WINDOW = 3          # slide_window
    # N_CLASS = 19        # class_num
    # n_filters = 1000      # num_filters
    # p_dropout = 0.5       # dropout rate
    # LR = 0.001     # learning_rate
    # DECAY = 5e-6
    # EPOCHS = 200
    # BATCH_SIZE = 32

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
                                        OPT=OPT
                                        )))

    # print('\n---------------------------------------------- Load Data -----------------------------------------------')

    data_train = pro.load_data('data/nine_train.txt')
    data_test = pro.load_data('data/nine_test.txt')

    target_dict = json.load(open('data/target_dict.txt', 'r', encoding='utf-8'))  # 19 classes
    c_target_dict = {value: key for key, value in target_dict.items()}

    tr_target_dict = json.load(open('data/tr_target_dict.txt', 'r', encoding='utf-8'))  # 10 classes, ignore direction

    '''
    word_dict: 19215 words and their id
    {'chop': 9137, 'negatives': 19215, 'bizarre': 6267, 'corners': 9138, 'flyers': 6268, 'belly': 2748}
    '''
    word_dict = pro.build_dict(data_train[0])

    x, y, dist1, dist2 = pro.vectorize(data_train, word_dict, MAX_LEN)
    y = np.array(y).astype(np.int64)
    np_cat = np.concatenate((x, np.array(dist1), np.array(dist2)), 1)

    e_x, e_y, e_dist1, e_dist2 = pro.vectorize(data_test, word_dict, MAX_LEN)
    y = np.array(y).astype(np.int64)
    eval_cat = np.concatenate((e_x, np.array(e_dist1), np.array(e_dist2)), 1)

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
        # transform y_true
        ay = list(y.cpu().data.numpy())
        c_y = [c_target_dict[item] for item in ay]      # c_target_dict: 19 relations
        y_true = np.array([tr_target_dict[item] for item in c_y])   # tr_target_dict: 10 relations

        # transform y_predict
        predict = torch.max(sc, 1)[1].long()
        ap = list(predict.cpu().data.numpy())
        c_p = [c_target_dict[item] for item in ap]
        y_predict = np.array([tr_target_dict[item] for item in c_p])
        f1 = f1_score(y_true, y_predict, average='micro')
        return f1 * 100

    best_score = 0
    patience = 0
    for i in range(1, N_EPOCHS + 1):
        patience += 1

        # train
        train = torch.from_numpy(np_cat.astype(np.int64))
        y_tensor = torch.LongTensor(y)
        train_datasets = D.TensorDataset(data_tensor=train, target_tensor=y_tensor)
        train_dataloader = D.DataLoader(train_datasets, BATCH_SIZE, True, num_workers=2)
        score_train = 0
        loss = 0
        n_trained_batch = 0
        for (batch_x_cat, batch_y) in train_dataloader:
            n_trained_batch += 1
            bx, bd1, bd2, by = data_unpack(batch_x_cat, batch_y)
            weight_o = model(bx, bd1, bd2)
            loss_per_batch = loss_func(weight_o, by)
            optimizer.zero_grad()
            loss_per_batch.backward()
            optimizer.step()
            loss += loss_per_batch
            score_train += prediction(weight_o, by)
        # end for
        loss = loss.cpu().data.numpy()[0] / n_trained_batch
        score_train = score_train / n_trained_batch

        # evaluate
        eval = torch.from_numpy(eval_cat.astype(np.int64))
        score_eval = 0
        n_eval_batch = 0
        y_tensor = torch.LongTensor(e_y)
        eval_datasets = D.TensorDataset(data_tensor=eval, target_tensor=y_tensor)
        eval_dataloader = D.DataLoader(eval_datasets, BATCH_SIZE, True, num_workers=2)
        for (batch_x_cat, batch_y) in eval_dataloader:
            bx, bd1, bd2, by = data_unpack(batch_x_cat, batch_y)
            wo = model(bx, bd1, bd2, False)
            score_eval += prediction(wo, by)
            n_eval_batch += 1
        # end for
        score_eval = score_eval / n_eval_batch
        # if i % 10 == 0:
        print('[{}/{}]\t train_loss: {:4f}\t train_f1: {:.3f}\t test_f1: {:.3f}'.format(i, N_EPOCHS, loss, score_train, score_eval))

        # save best model
        current_score = score_eval
        if current_score > best_score:
            patience = 0
            best_score = current_score
            torch.save(model.state_dict(), fpath_best_model)
            print('Model saved to <{}>'.format(fpath_best_model))

        if patience >= 20:
            print('Earlystopping: patience = {}'.format(patience))
            break
        # end for
    # eval = torch.from_numpy(np_cat.astype(np.int64))
    # model.load_state_dict(torch.load(fpath_saved_model))

    with open('saved_models/20190118/results.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([TIMESTAMP, round(best_score, 3), OPT, BATCH_SIZE, N_EPOCHS, LR, DECAY, n_filters, p_dropout, MAX_LEN, POS_EMBED_SIZE, N_POS_EMBED, WINDOW, N_CLASS])
    f.close()

    print('\n------------------------------------------------- END --------------------------------------------------\n\n\n')

# end def


if __name__ == '__main__':
    # create_csv('results.csv')
    main()
