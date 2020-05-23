import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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

    shipname = df_general.index
    shipname2 = df_machine.index
    subsections_g = [str(x) for x in df_general.columns]
    subsections_m = [str(x) for x in df_machine.columns]


    # print('GENERAL PART\n', df_general)
    # print('SHIP NAME GENERAL:', shipname)
    # print('SHIP NAME MACHINERY:', shipname2)
    print('CHECK SHIP NAMES ARE SAME: ', end='')
    print((shipname == shipname2[:-1]).all())
    # print('SUBSECTION GENERAL', subsections_g)
    # print('SUBSECTION MACHINERY', subsections_m)

    dic = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5,
           'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11}

    train_score_array = []
    test_score_array = []
    cross_val_array = []
    xrange = []
    for i in range(len(subsections_m)):
        num_label = int(df_machine.iloc[-1][i])
        if num_label > 1:
            print('\nPREDICT', subsections_m[i])
            print('num of labels:', num_label)
            x = []
            y = []
            for j in range(len(shipname)):
                x.append([dic[x] for x in df_general.iloc[j]])
                y.append(dic[df_machine.iloc[j][i]])
            X = np.array(x)
            Y = np.array(y)

            # print(X)
            # print(Y)

            # model = KNeighborsClassifier(n_neighbors=num_label)
            model = LinearSVC(loss='hinge', C=0.1, max_iter=20000)
            # model = MLPClassifier(solver="sgd",
            #                       random_state=0,
            #                       hidden_layer_sizes=[30],
            #                       max_iter=10000)


            # split train and test data
            x_train, x_test, y_train, y_test = train_test_split(X,
                                                                Y,
                                                                test_size=0.3,
                                                                random_state=0)
            model.fit(x_train, y_train)
            score_tr = model.score(x_train, y_train)
            score_te = model.score(x_test, y_test)
            print("train score:", score_tr)
            print("test score:", score_te)
            # print(model.coef_)
            #
            train_score_array.append(score_tr)
            test_score_array.append(score_te)


            # Cross Validation Score
            stratifiedkfold = StratifiedKFold(n_splits=3)
            scores = cross_val_score(model, X, Y, cv=stratifiedkfold)
            print('cross validation score: {}'.format(np.mean(scores)))
            cross_val_array.append(np.mean(scores))

            # grid_search(X, Y)


            xrange.append(subsections_m[i])


    fig, ax = plt.subplots()
    ax.set_xlabel('SUBSECTIONS')
    ax.set_ylabel('SCORE')
    ax.plot(xrange, train_score_array, label='train', alpha=0.3)
    ax.plot(xrange, test_score_array, label='test', alpha=1)
    ax.plot(xrange, cross_val_array, label='cross validation', alpha=0.3)
    ax.legend(loc='lower center')
    ax.set_xticklabels(xrange, rotation=270, ha='left')
    ax.set_ylim(0.2, 1)
    ax.grid(axis='y', c='gainsboro')
    fig.tight_layout()
    os.makedirs('./fig/ML_SCORE', exist_ok=True)
    plt.savefig('./fig/ML_SCORE/svc_c0.1_score.png')



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