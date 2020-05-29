import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import os
import shutil
import TextProcess

import networkx as nx
from scipy import stats as st
import seaborn as sns

import math



def CreateDSM(path='./excel/Machinery_Label_Matrix.xlsx',
              savename='Machinery'):

    print('Create DSM of', savename)

    df = pd.read_excel(path, sheet_name=0, header=0, index_col=0)
    subsections = [str(x) for x in df.columns]
    num_subsec = len(subsections)

    DSM = np.zeros((num_subsec, num_subsec), dtype='int8')
    weight_Array = []
    criterion = len(df)-1

    fig_path = './fig/Heatmap/' + savename + '/'
    if os.path.exists(fig_path):
        shutil.rmtree(fig_path)
    os.makedirs(fig_path)

    for i in range(num_subsec):
        DSM[i][i] = 1
        for j in range(i+1, num_subsec):
            label0 = df.iloc[:-1, i]
            label1 = df.iloc[:-1, j]

            # num of types of labels
            num0 = int(df.iloc[-1, i])
            num1 = int(df.iloc[-1, j])
            if (num0 > 1) and (num1 > 1):
                # create cross-tab table
                cross = pd.crosstab(label0, label1)
                # chi-square test
                chi2, p, dof, ef = st.chi2_contingency(cross, correction=False)

                if p < 1.0e-5:
                    DSM[i][j] = 1
                    DSM[j][i] = 1

                    # prepare to make npy file
                    namei = subsections[i]
                    namej = subsections[j]
                    weight_Array.append((namei, namej))


                    # make heatmap figure
                    name = namei + '&' + namej
                    fig, ax = plt.subplots(figsize=(8, 8))
                    sns.heatmap(data=cross, cmap="RdBu_r", annot=True, square=True)
                    ax.set_title('p: {0:.2e}\nchi2: {1}'.format(p, chi2))
                    plt.savefig(fig_path + name + '.png')
                    print(fig_path + name + '.png is Saved')
                    plt.close()



    value = [[''] + subsections]
    for i in range(num_subsec):
        value.append([subsections[i]] + list(DSM[i]))

    weight_Array = np.array(weight_Array)
    dir = './excel/DSM/'
    os.makedirs(dir, exist_ok=True)
    np.save(dir + savename + '_weight.npy', weight_Array)

    sheet_name = 'DSM'
    TextProcess.write_matrix_excel_xlsx(dir + savename + '_DSM.xlsx',
                                        sheet_name, value)

    print(dir + savename + '_DSM.xlsx Writing Succeed!')


def CreateNetwork(savename='Machinery'):

    print('Create Network')
    weighted_edges = np.load('./excel/DSM/' + savename + '_weight.npy',
                             allow_pickle=True)

    G = nx.Graph()
    G.add_edges_from(weighted_edges)
    # print(G.edges(data=True))

    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G)
    color = [_SelectEdgeColor(u, v) for (u, v, d) in G.edges(data=True)]
    nx.draw_networkx_nodes(G, pos, node_color='b', alpha=0.4)  # node_size=size
    nx.draw_networkx_edges(G, pos, edge_color=color, alpha=0.4)
    nx.draw_networkx_labels(G, pos, fontsize=6)

    plt.axis('off')

    save_dir = './fig/NETWORK/'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_dir + savename + '_network.png')

    print(save_dir + savename + '_network.png Writing Succeed!')


def _SelectEdgeColor(u, v):
    if u[:2] == v[:2]:
        return 'gray'
    else:
        return 'red'


def _SelectEdgeColor2(u, v):
    if u[:2] == v[:2]:
        return 'gray'
    elif u[0] != v[0]:
        return 'red'
    else:
        return 'gray'


def connect_xlsx(path_g='./excel/General_Label_Matrix.xlsx',
                 path_m='./excel/Machinery_Label_Matrix.xlsx',
                 save_path='./excel/Generanl&Machinery_Label_Matrix.xlsx'):

    print('{0} & {1} Connecting...'.format(path_g, path_m))

    dfG = pd.read_excel(path_g, sheet_name=0, header=0, index_col=0)
    dfM = pd.read_excel(path_m, sheet_name=0, header=0, index_col=0)

    dfG.rename(columns=lambda chapter: 'G'+chapter, inplace=True)
    dfM.rename(columns=lambda chapter: 'M'+chapter, inplace=True)

    df_connect = pd.concat([dfG, dfM], axis=1)

    df_connect.to_excel(save_path)
    print('{0} Writing Succeed!'.format(save_path))


def difference_dsm(path='./excel/Machinery_Label_Matrix.xlsx',
                   savename='MachineryD'):

    print('Create DSM of', savename)

    df = pd.read_excel(path, sheet_name=0, header=0, index_col=0)
    subsections = [str(x) for x in df.columns]
    num_ships = len(df) - 1
    num_subsec = len(subsections)

    DSM = np.zeros((num_subsec, num_subsec), dtype='float64')
    tf_matrix = []
    count = 0

    for i in range(num_ships):
        for j in range(i+1, num_ships):
            label0 = df.iloc[i]
            label1 = df.iloc[j]

            tf_series = (label0 != label1)
            tf_matrix.append(tf_series)
            count += 1

    tf_matrix = np.array(tf_matrix)
    count_array = np.count_nonzero(tf_matrix, axis=0) / count
    # print(count_array)

    for ship in range(num_ships):
        for i in range(num_subsec):
            for j in range(i+1, num_subsec):
                if tf_matrix[ship][i] and tf_matrix[ship][j]:
                    DSM[i][j] += 1  # (1 - count_array[i])*(1 - count_array[j])
                    DSM[j][i] += 1  # (1 - count_array[i])*(1 - count_array[j])

    df = pd.DataFrame(DSM, columns=subsections, index=subsections)

    dir = './excel/DSM/'
    os.makedirs(dir, exist_ok=True)

    sheet_name = 'DSM'
    df.to_excel(dir + savename + '_DSM.xlsx', sheet_name)
    print(dir + savename + '_DSM.xlsx Writing Succeed!')


def CreateWeightedNetwork(savename='MachineryD'):

    print('Create Network')
    path = './excel/DSM/' + savename + '_DSM.xlsx'
    df = pd.read_excel(path, sheet_name=0, header=0, index_col=0)
    subsections = [str(x) for x in df.index]
    n = len(subsections)
    edge_list = []
    for i in range(n):
        for j in range(i+1, n):
            value = df.iloc[i][j]
            if value >= 60:
                edge_list.append([subsections[i], subsections[j], value])

    G = nx.Graph()
    G.add_weighted_edges_from(edge_list)

    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G)
    width = [(d["weight"]-50) * 0.08 for (u, v, d) in G.edges(data=True)]
    nx.draw_networkx_nodes(G, pos, node_color='b', alpha=0.4)  # node_size=size
    nx.draw_networkx_edges(G, pos, edge_color='r', alpha=0.2, width=width)
    nx.draw_networkx_labels(G, pos, fontsize=6)

    plt.axis('off')

    save_dir = './fig/NETWORK/'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_dir + savename + '_network.png')

    print(save_dir + savename + '_network.png Writing Succeed!')


if __name__ == '__main__':
    # CreateDSM()
    # CreateNetwork()

    # connect_xlsx()
    # CreateDSM(path='./excel/Generanl&Machinery_Label_Matrix.xlsx', savename='Connect')
    # CreateNetwork(savename='Connect')

    # CreateDSM(path='./excel/General_Label_Matrix.xlsx', savename='General')
    # CreateNetwork(savename='General')

    # difference_dsm()
    # CreateWeightedNetwork()

    # difference_dsm(path='./excel/Generanl&Machinery_Label_Matrix.xlsx', savename='ConnectD')
    # CreateNetwork(savename='ConnectD')

    # difference_dsm(path='./excel/General_Label_Matrix.xlsx', savename='GeneralD')
    CreateWeightedNetwork(savename='GeneralD')