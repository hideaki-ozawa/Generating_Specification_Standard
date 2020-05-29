import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def machine_learning(general_xlsx_path='./excel/General_Label_Matrix.xlsx',
                     machinery_xlsx_path='./excel/Machinery_Label_Matrix.xlsx'):

    df_general = pd.read_excel(general_xlsx_path, sheet_name=0, header=0, index_col=0)
    df_machine = pd.read_excel(machinery_xlsx_path, sheet_name=0, header=0, index_col=0)

    shipname = df_general[:-1].index
    shipname2 = df_machine.index
    subsections_g = [str(x) for x in df_general.columns]
    subsections_m = [str(x) for x in df_machine.columns]
    num_ship = len(shipname)


    # print('GENERAL PART\n', df_general)
    # print('SHIP NAME GENERAL:', shipname)
    # print('SHIP NAME MACHINERY:', shipname2)
    print('CHECK SHIP NAMES ARE SAME: ', end='')
    print((shipname == shipname2[:-1]).all())
    # print('SUBSECTION GENERAL', subsections_g)
    # print('SUBSECTION MACHINERY', subsections_m)

    dic = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4,
           'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10}
    rdic = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
           5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K'}

    train_score_array = []
    test_score_array = []
    cross_val_array = []
    xrange = []
    X = []
    y = []
    for i in range(num_ship):
        X.append(_one_hot_encoding(df_general.iloc[i], dic))
    X = np.array(X)
    X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
    print(X.shape)

    for i in range(len(subsections_m)):
        num_label = int(df_machine.iloc[-1][i])
        if num_label > 1:
            try:
                name = subsections_m[i]
                print('\nPREDICT', name)
                print('num of labels:', num_label)
                y = []
                for j in range(len(shipname)):
                    y.append(dic[df_machine.iloc[j][i]])
                y = np.array(y)
                freq, bar = np.histogram(y, range=(0, 10))
                print('freq: ', freq)
                print('bar: ', bar)

                # print(X)
                # print(Y)

                # model = KNeighborsClassifier(n_neighbors=num_label)
                model = LinearSVC(loss='hinge', C=0.1, max_iter=20000)
                # model = MLPClassifier(solver="sgd",
                #                       random_state=0,
                #                       hidden_layer_sizes=[30],
                #                       max_iter=10000)


                # split train and test data
                X_train, X_test, y_train, y_test = train_test_split(X,
                                                                    y,
                                                                    test_size=0.3,
                                                                    random_state=0)
                model.fit(X_train, y_train)
                score_tr = model.score(X_train, y_train)
                score_te = model.score(X_test, y_test)
                print("train score:", score_tr)
                print("test score:", score_te)

                weight = model.coef_
                weight = weight.reshape(weight.shape[0], 26, 10)
                fig = plt.figure(figsize=(8, 8))
                fig.suptitle(name + ' (score: {0:.2f})'.format(score_te))
                for i in range(weight.shape[0]):
                    _heatmap(weight, i, fig, subsections_g)
                    text = '{0} (n={1})\n{2:.1f}%'.format(rdic[i], freq[i], freq[i]*100/num_ship)
                    fig.text(0.88, 0.90-0.16*i, text)
                    if weight.shape[0] == 1:
                        text = '{0} (n={1})\n{2:.1f}%'.format(rdic[1], freq[1], freq[1]*100/num_ship)
                        fig.text(0.88, 0.82, text)
                if score_te >= 0.8:
                    save_dir = './fig/coef/true/'
                else:
                    save_dir = './fig/coef/false/'
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(save_dir + name + '.png')
                print(save_dir + name + '.png' + '.png is Saved')
                plt.close()


                train_score_array.append(score_tr)
                test_score_array.append(score_te)


                # Cross Validation Score
                stratifiedkfold = StratifiedKFold(n_splits=3)
                scores = cross_val_score(model, X, y, cv=stratifiedkfold)
                print('cross validation score: {}'.format(np.mean(scores)))
                cross_val_array.append(np.mean(scores))

                # grid_search(X, Y)


                xrange.append(name)
            except ValueError as e:
                print(e)


    fig = plt.figure()
    ax0 = fig.add_axes((0.1, 0.1, 0.7, 0.8))
    ax1 = fig.add_axes((0.85, 0.1, 0.08, 0.8), sharey=ax0)
    ax0.set_xlabel('SUBSECTIONS')
    ax0.set_ylabel('SCORE')
    ax0.plot(xrange, train_score_array, label='train', alpha=0.3)
    ax0.plot(xrange, test_score_array, label='test', alpha=1)
    ax0.plot(xrange, cross_val_array, label='cross validation', alpha=0.3)
    ax0.legend(loc='lower center')
    ax0.set_xticklabels(xrange, rotation=270, ha='left')
    ax0.set_ylim(0.2, 1)
    ax0.grid(axis='y', c='gainsboro')
    ax1.boxplot(test_score_array, whis=[0, 100])
    ax1.tick_params(labelleft=False, labelbottom=False)
    os.makedirs('./fig/ML_SCORE', exist_ok=True)
    plt.savefig('./fig/ML_SCORE/svc_c0.1_score.png')
    plt.close()
    print('svc_c0.1_score.png Writing Succeed')


def _one_hot_encoding(row, dic):
    '''
    :param x: ['B', 'A', 'K', 'A', ..., 'B']
    :return:   [[0, 1, 0, 1, ..., 0],
                [1, 0, 0, 0, ..., 1],
                 :  :  :  :       :
                [0, 0, 1, 0, ..., 0]]
    '''

    length = len(row)
    nparray = np.zeros((length, 10))
    for i, s in enumerate(row):
        nparray[i][dic[s]] = 1

    return nparray


def _heatmap(weight, i, fig, subsections_g):
    bottom = 0.84 - 0.16*i
    ax = fig.add_axes([0.05, bottom, 0.8, 0.12])
    # cbar_ax = fig.add_axes([0.9, 0.6, 0.03, 0.3])
    data = weight[i]
    data = data.T
    ylabel = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    sns.heatmap(data=data, ax=ax,
                cmap="RdBu_r", cbar=None,
                annot=False, square=False,
                vmin=-0.3, vmax=0.3,
                yticklabels=ylabel
                )
    ax.set_xticklabels(subsections_g, size=8)


def grid_search(train_features, train_labels):
    loss_list = ['hinge', 'squared_hinge']
    param_list = [0.001, 0.01, 0.1, 1, 10, 100]
    best_score = 0
    best_parameters = {}

    for loss in loss_list:
        for C in param_list:
            svm = LinearSVC(loss=loss, C=C)
            scores = cross_val_score(svm, train_features, train_labels, cv=5)
            score = np.mean(scores)
            if score > best_score:
                best_score = score
                best_parameters = {'loss': loss, 'C': C}


    print('Best score on validation set: {}'.format(best_score))
    print('Best parameters: {}'.format(best_parameters))




if __name__ == '__main__':
    machine_learning()