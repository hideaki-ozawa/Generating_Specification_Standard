import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import os
import shutil
import TextProcess

import networkx as nx
from scipy.stats import chi2_contingency
import seaborn as sns



def CreateDSM(path='./excel/label list0.xlsx',
              save_path='./excel/DSM.xlsx'):

    df = pd.read_excel(path, sheet_name=0, header=0, index_col=0)
    subsections = [str(x) for x in df.columns]
    num_subsec = len(subsections)

    DSM = np.zeros((num_subsec, num_subsec), dtype='int8')
    weight_Array = []
    criterion = len(df)-1

    if os.path.exists('./fig/Heatmap'):
        shutil.rmtree('./fig/Heatmap')
    os.mkdir('./fig/Heatmap')

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
                chi2, p, dof, ef = chi2_contingency(cross, correction=False)

                if chi2 > criterion:
                    DSM[i][j] = 1
                    DSM[j][i] = 1

                    namei = subsections[i]
                    namej = subsections[j]
                    # weight = float(-np.log(p)/10)
                    # weight = float(chi2-criterion)
                    weight_Array.append((namei, namej))


                    # make heatmap figure
                    name = namei + '&' + namej
                    fig, ax = plt.subplots(figsize=(8, 8))
                    sns.heatmap(data=cross, cmap="RdBu_r", annot=True, square=True)
                    ax.set_title('p: {0:.2e}\nchi2: {1}'.format(p, chi2))
                    plt.savefig('./fig/Heatmap/' + name + '.png')
                    plt.close()



    value = [[''] + subsections]
    for i in range(num_subsec):
        value.append([subsections[i]] + list(DSM[i]))

    weight_Array = np.array(weight_Array)
    os.makedirs('./fig/NETWORK', exist_ok=True)
    np.save('./fig/NETWORK/weightArray.npy', weight_Array)

    sheet_name = 'DSM'
    TextProcess.write_matrix_excel_xlsx(save_path, sheet_name, value)



def CreateNetwork(path='./excel/DSM.xlsx'):

    # df = pd.read_excel(path, sheet_name=0, header=0, index_col=0)
    # subsections = [str(x) for x in df.index]
    # n = len(subsections)
    #
    # edge_list = []
    # for i in range(n):
    #     for j in range(i+1, n):
    #         if df.iloc[i][j] == 1:
    #             edge_list.append([subsections[i], subsections[j]])
    weighted_edges = np.load('./fig/NETWORK/weightArray.npy', allow_pickle=True)

    G = nx.Graph()
    # G.add_edges_from(edge_list)
    G.add_weighted_edges_from(weighted_edges, weight='edge_weight')
    # print(G.edges(data=True))

    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, k=1.0)
    # pr = nx.pagerank(G)
    # size = [10000*v for v in pr.values()]
    color = [_SelectEdgeColor(u, v) for (u, v, d) in G.edges(data=True)]
    nx.draw_networkx_nodes(G, pos, node_color='b', alpha=0.4)  # node_size=size
    nx.draw_networkx_edges(G, pos, edge_color=color, alpha=0.4)
    nx.draw_networkx_labels(G, pos, fontsize=10)

    plt.axis('off')

    os.makedirs('./fig/NETWORK', exist_ok=True)
    plt.savefig('./fig/NETWORK/network_machinery.png')


def _SelectEdgeColor(u, v):
    if u[:2] == v[:2]:
        return 'gray'
    else:
        return 'red'


if __name__ == '__main__':
    # CreateDSM()
    CreateNetwork()