import TextProcess
from TextProcess import subsection_value
import Cluster

import shutil
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
from scipy.cluster.hierarchy import dendrogram, linkage

abbreviation = ['reg', 'i.e', 'no', 'incl', 'vessel', 'res', 'rev', 'msc', 'f.o', 'pty', 'ltd', 'code',
                'inc', 'abt', 'i.c.t.m', 'm.c.r', 'dwg', 'w.b.t', 'nos', 'm.t']
txt_path = './txt/General_Part/section'
pdf_path = TextProcess.get_pdf_path('./pdf/pdf_path')
shipname = [None]*len(pdf_path)
for i in range(0, len(pdf_path)):
    shipname[i] = pdf_path[i][-13:-9]


def preprocess_general():
    """
    Put the pdf file in './pdf/input'. Get corresponding txt file(general part) in './pdf/output1'.
    """
    shutil.rmtree('./pdf/output1')
    os.mkdir('./pdf/output1')

    pdf_path = TextProcess.get_pdf_path('./pdf/input')

    for i in range(0, len(pdf_path)):
        TextProcess.split_pdf(4, 31, pdf_path[i], './pdf/output0/' + pdf_path[i][-14:])
    pdf_path = TextProcess.get_pdf_path('./pdf/output0')
    for i in range(0, len(pdf_path)):
        TextProcess.convert_pdf_to_txt(pdf_path[i],
                                   './pdf/output1/' + pdf_path[i][-14:-4] + '.txt')
    txt_path = TextProcess.get_txt_path('./pdf/output1')
    for i in range(0, len(txt_path)):
        TextProcess.remove_n(txt_path[i])

    shutil.rmtree('./pdf/output0')
    os.mkdir('./pdf/output0')
    shutil.rmtree('./pdf/input')
    os.mkdir('./pdf/input')


def preprocess_hull():
    """
    Put the pdf file in './pdf/input'. Get corresponding txt file(hull part) in './pdf/output1'.
    """
    shutil.rmtree('./pdf/output1')
    os.mkdir('./pdf/output1')

    pdf_path = TextProcess.get_pdf_path('./pdf/input')

    for i in range(0, len(pdf_path)):
        TextProcess.split_pdf_endpage(34, pdf_path[i], './pdf/output0/' + pdf_path[i][-14:])
    pdf_path = TextProcess.get_pdf_path('./pdf/output0')
    for i in range(0, len(pdf_path)):
        TextProcess.convert_pdf_to_txt(pdf_path[i],
                                   './pdf/output1/' + pdf_path[i][-14:-4] + '.txt')
    txt_path = TextProcess.get_txt_path('./pdf/output1')
    for i in range(0, len(txt_path)):
        TextProcess.remove_n(txt_path[i])

    shutil.rmtree('./pdf/output0')
    os.mkdir('./pdf/output0')
    shutil.rmtree('./pdf/input')
    os.mkdir('./pdf/input')


def write_difference_to_excel():
    """
    Find the difference between specifications and union set and write it into xlsx file.
    """

    book_name_xlsx = './excel/General_Part/Section1.xlsx'
    sheet_name_xlsx = ['1.1', '1.2']
    value_1 = [subsection_value(shipname, '_1_1', abbreviation, txt_path),
               subsection_value(shipname, '_1_2', abbreviation, txt_path)]
    TextProcess.write_excel_xlsx(book_name_xlsx, sheet_name_xlsx, value_1)

    book_name_xlsx = './excel/General_Part/Section2.xlsx'
    sheet_name_xlsx = ['2.1', '2.2', '2.3']
    value_2 = [subsection_value(shipname, '_2_1', abbreviation, txt_path),
               subsection_value(shipname, '_2_2', abbreviation, txt_path),
               subsection_value(shipname, '_2_3', abbreviation, txt_path)]
    TextProcess.write_excel_xlsx(book_name_xlsx, sheet_name_xlsx, value_2)

    book_name_xlsx = './excel/General_Part/Section3.xlsx'
    sheet_name_xlsx = ['3.1', '3.2', '3.3', '3.4', '3.5']
    value_3 = [subsection_value(shipname, '_3_1', abbreviation, txt_path),
               subsection_value(shipname, '_3_2', abbreviation, txt_path),
               subsection_value(shipname, '_3_3', abbreviation, txt_path),
               subsection_value(shipname, '_3_4', abbreviation, txt_path),
               subsection_value(shipname, '_3_5', abbreviation, txt_path)]
    TextProcess.write_excel_xlsx(book_name_xlsx, sheet_name_xlsx, value_3)

    # book_name_xlsx = './excel/General_Part/Section4.xlsx'
    # sheet_name_xlsx = ['4']
    # value_4 = [subsection_value(shipname, '_4_0', abbreviation, txt_path)]
    # TextProcess.write_excel_xlsx(book_name_xlsx, sheet_name_xlsx, value_4)

    book_name_xlsx = './excel/General_Part/Section5.xlsx'
    sheet_name_xlsx = ['5.1', '5.2', '5.3', '5.4', '5.5', '5.6']
    value_5 = [subsection_value(shipname, '_5_1', abbreviation, txt_path),
               subsection_value(shipname, '_5_2', abbreviation, txt_path),
               subsection_value(shipname, '_5_3', abbreviation, txt_path),
               subsection_value(shipname, '_5_4', abbreviation, txt_path),
               subsection_value(shipname, '_5_5', abbreviation, txt_path),
               subsection_value(shipname, '_5_6', abbreviation, txt_path)]
    TextProcess.write_excel_xlsx(book_name_xlsx, sheet_name_xlsx, value_5)

    book_name_xlsx = './excel/General_Part/Section5.xlsx'
    sheet_name_xlsx = ['5.1', '5.2', '5.3', '5.4', '5.5', '5.6']
    value_5 = [subsection_value(shipname, '_5_1', abbreviation, txt_path),
               subsection_value(shipname, '_5_2', abbreviation, txt_path),
               subsection_value(shipname, '_5_3', abbreviation, txt_path),
               subsection_value(shipname, '_5_4', abbreviation, txt_path),
               subsection_value(shipname, '_5_5', abbreviation, txt_path),
               subsection_value(shipname, '_5_6', abbreviation, txt_path)]
    TextProcess.write_excel_xlsx(book_name_xlsx, sheet_name_xlsx, value_5)

    book_name_xlsx = './excel/General_Part/Section6.xlsx'
    sheet_name_xlsx = ['6.1', '6.2', '6.3']
    value_6 = [subsection_value(shipname, '_6_1', abbreviation, txt_path),
               subsection_value(shipname, '_6_2', abbreviation, txt_path),
               subsection_value(shipname, '_6_3', abbreviation, txt_path)]
    TextProcess.write_excel_xlsx(book_name_xlsx, sheet_name_xlsx, value_6)

    book_name_xlsx = './excel/General_Part/Section7.xlsx'
    sheet_name_xlsx = ['7.1', '7.2', '7.3', '7.4', '7.5', ]
    value_7 = [subsection_value(shipname, '_7_1', abbreviation, txt_path),
               subsection_value(shipname, '_7_2', abbreviation, txt_path),
               subsection_value(shipname, '_7_3', abbreviation, txt_path),
               subsection_value(shipname, '_7_4', abbreviation, txt_path),
               subsection_value(shipname, '_7_5', abbreviation, txt_path), ]
    TextProcess.write_excel_xlsx(book_name_xlsx, sheet_name_xlsx, value_7)

    book_name_xlsx = './excel/General_Part/Section8.xlsx'
    sheet_name_xlsx = ['8']
    value_8 = [subsection_value(shipname, '_8_1', abbreviation, txt_path)]
    TextProcess.write_excel_xlsx(book_name_xlsx, sheet_name_xlsx, value_8)

    book_name_xlsx = './excel/General_Part/Section9.xlsx'
    sheet_name_xlsx = ['9']
    value_9 = [subsection_value(shipname, '_9_1', abbreviation, txt_path)]
    TextProcess.write_excel_xlsx(book_name_xlsx, sheet_name_xlsx, value_9)


def cos_distance_dendrogram():
    """
    Calculate the cosine distance between any 2 of the specifications and write the matrix into xlsx file.
    Draw the dendrogram with the matrix.
    """
    subsection = ['_1_1', '_1_2']
    matrix_1 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_2_1', '_2_2', '_2_3']
    matrix_2 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_3_1', '_3_2', '_3_3', '_3_4', '_3_5']
    matrix_3 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_5_1', '_5_2', '_5_3', '_5_4', '_5_5', '_5_6']
    matrix_5 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_6_1', '_6_2', '_6_3']
    matrix_6 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_7_1', '_7_2', '_7_3', '_7_4', '_7_5']
    matrix_7 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_8_1']
    matrix_8 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_9_1']
    matrix_9 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)

    matrix = []
    for i in range(0, len(shipname)):
        matrix.append(
            matrix_1[i] + matrix_2[i] + matrix_3[i] + matrix_5[i] + matrix_6[i] + matrix_7[i] + matrix_8[i] + matrix_9[
                i])

    matrix = np.array(matrix)
    distance_matrix = TextProcess.cos_distance(matrix)

    rows = len(shipname) + 1
    cols = len(shipname) + 1
    value_distance = [[0] * cols for i in range(rows)]

    fullname = [None]*len(shipname)
    for i in range(0, len(shipname)):
        fullname[i] = 'S' + shipname[i] + 'C9000'

    value_distance[0][0] = ''
    for i in range(1, cols):
        value_distance[0][i] = fullname[i - 1]
    for i in range(1, rows):
        value_distance[i][0] = fullname[i - 1]
    for i in range(1, rows):
        for j in range(1, cols):
            value_distance[i][j] = distance_matrix[i - 1][j - 1]

    book_name_xlsx = './excel/Cosine Similarity.xlsx'
    sheet_name_xlsx = 'Cosine Similarity'
    TextProcess.write_matrix_excel_xlsx(book_name_xlsx, sheet_name_xlsx, value_distance)

    distance_matrix = spatial.distance.pdist(matrix, 'cosine')
    Z = linkage(distance_matrix, 'ward')
    labelList = shipname
    dendrogram(Z, labels=labelList)

    plt.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.10)

    plt.savefig('./dendrogram/cos/All.png')
    plt.cla()

    matrix_list = [matrix_1,
                   matrix_2,
                   matrix_3,
                   matrix_5,
                   matrix_6,
                   matrix_7,
                   matrix_8,
                   matrix_9]
    section_name = ['1', '2', '3', '5', '6', '7', '8', '9']

    for i in range(0, len(matrix_list)):
        matrix = matrix_list[i]
        matrix = np.array(matrix)
        distance_matrix = spatial.distance.pdist(matrix, 'cosine')
        Z = linkage(distance_matrix, 'ward')
        labelList = shipname
        dendrogram(Z, labels=labelList)
        plt.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.10)
        plt.savefig('./dendrogram/cos/Section' + section_name[i] + '.png')
        plt.cla()


def num_difference_dendrogram():
    """
    Calculate the number of different sentences between any 2 of the specifications and write the matrix into xlsx file.
    Draw the dendrogram with the matrix.
    """
    subsection = ['_1_1', '_1_2']
    matrix_1 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_2_1', '_2_2', '_2_3']
    matrix_2 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_3_1', '_3_2', '_3_3', '_3_4', '_3_5']
    matrix_3 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_5_1', '_5_2', '_5_3', '_5_4', '_5_5', '_5_6']
    matrix_5 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_6_1', '_6_2', '_6_3']
    matrix_6 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_7_1', '_7_2', '_7_3', '_7_4', '_7_5']
    matrix_7 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_8_1']
    matrix_8 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_9_1']
    matrix_9 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)

    matrix = []
    for i in range(0, len(shipname)):
        matrix.append(
            matrix_1[i] + matrix_2[i] + matrix_3[i] + matrix_5[i] + matrix_6[i] + matrix_7[i] + matrix_8[i] + matrix_9[
                i])

    matrix = np.array(matrix)
    difference = TextProcess.num_difference_matrix(matrix)

    rows = len(shipname) + 1
    cols = len(shipname) + 1
    value_distance = [[0] * cols for i in range(rows)]

    fullname = [None] * len(shipname)
    for i in range(0, len(shipname)):
        fullname[i] = 'S' + shipname[i] + 'C9000'

    value_distance[0][0] = ''
    for i in range(1, cols):
        value_distance[0][i] = fullname[i - 1]
    for i in range(1, rows):
        value_distance[i][0] = fullname[i - 1]
    for i in range(1, rows):
        for j in range(1, cols):
            value_distance[i][j] = difference[i - 1][j - 1]

    book_name_xlsx = './excel/#difference.xlsx'
    sheet_name_xlsx = '#difference'
    TextProcess.write_matrix_excel_xlsx(book_name_xlsx, sheet_name_xlsx, value_distance)

    difference = TextProcess.squareform(difference)
    difference = difference / len(matrix[0])

    Z = linkage(difference, 'ward')
    dendrogram(Z, labels=shipname)

    plt.subplots_adjust(left=0.07, right=0.98, top=0.98, bottom=0.09)
    plt.savefig('./dendrogram/num/All.png')
    plt.cla()

def common_and_different_items():
    """
    Find the common sentences at each node(common sentences in two clusters) and different sentences between a node and
    its sub-node in the dendrogram. Write them into xlsx files.
    """
    subsection = ['_1_1', '_1_2']
    (matrix_1, item_union1) = TextProcess.section_matrix_and_item_union(shipname, subsection, abbreviation, txt_path)
    subsection = ['_2_1', '_2_2', '_2_3']
    (matrix_2, item_union2) = TextProcess.section_matrix_and_item_union(shipname, subsection, abbreviation, txt_path)
    subsection = ['_3_1', '_3_2', '_3_3', '_3_4', '_3_5']
    (matrix_3, item_union3) = TextProcess.section_matrix_and_item_union(shipname, subsection, abbreviation, txt_path)
    subsection = ['_5_1', '_5_2', '_5_3', '_5_4', '_5_5', '_5_6']
    (matrix_5, item_union5) = TextProcess.section_matrix_and_item_union(shipname, subsection, abbreviation, txt_path)
    subsection = ['_6_1', '_6_2', '_6_3']
    (matrix_6, item_union6) = TextProcess.section_matrix_and_item_union(shipname, subsection, abbreviation, txt_path)
    subsection = ['_7_1', '_7_2', '_7_3', '_7_4', '_7_5']
    (matrix_7, item_union7) = TextProcess.section_matrix_and_item_union(shipname, subsection, abbreviation, txt_path)
    subsection = ['_8_1']
    (matrix_8, item_union8) = TextProcess.section_matrix_and_item_union(shipname, subsection, abbreviation, txt_path)
    subsection = ['_9_1']
    (matrix_9, item_union9) = TextProcess.section_matrix_and_item_union(shipname, subsection, abbreviation, txt_path)
    section_matrix = [matrix_1, matrix_2, matrix_3, matrix_5, matrix_6, matrix_7, matrix_8, matrix_9]
    item_union = [item_union1, item_union2, item_union3, item_union5, item_union6, item_union7, item_union8,
                  item_union9]

    matrix = []
    for i in range(0, len(shipname)):
        matrix.append(
            matrix_1[i] + matrix_2[i] + matrix_3[i] + matrix_5[i] + matrix_6[i] + matrix_7[i] + matrix_8[i] + matrix_9[
                i])

    matrix = np.array(matrix)
    distance_matrix = spatial.distance.pdist(matrix, 'cosine')
    Z = linkage(distance_matrix, 'ward')
    Z1 = Z.astype(int)
    ship_matrixs = Cluster.section_matrix2ship_matrix(section_matrix)
    node_name = Cluster.node_name(shipname, Z1)
    node_matrixs = Cluster.node_matrixs(ship_matrixs, Z1)

    sheet_name_xlsx = ['Section1', 'Section2', 'Section3', 'Section5', 'Section6', 'Section7', 'Section8', 'Section9']

    for i in range(0, len(node_name)):
        value = [None] * len(sheet_name_xlsx)
        for j in range(0, len(sheet_name_xlsx)):
            value[j] = [Cluster.common_item(node_matrixs[i][j], item_union[j])]
        book_name_xlsx = './excel/Common_Item/' + node_name[i] + '.xlsx'
        TextProcess.write_excel_xlsx(book_name_xlsx, sheet_name_xlsx, value)

    node_matrixs_difference_L = Cluster.node_matrixs_difference(ship_matrixs, node_matrixs, Z1, 'L')
    node_matrixs_difference_R = Cluster.node_matrixs_difference(ship_matrixs, node_matrixs, Z1, 'R')

    for i in range(0, len(node_name)):
        value = [None] * len(sheet_name_xlsx)
        for j in range(0, len(sheet_name_xlsx)):
            value[j] = [Cluster.different_item(node_matrixs_difference_L[i][j], item_union[j])]
        book_name_xlsx = './excel/Different_Item/' + node_name[i] + '_left.xlsx'
        TextProcess.write_excel_xlsx(book_name_xlsx, sheet_name_xlsx, value)

    for i in range(0, len(node_name)):
        value = [None] * len(sheet_name_xlsx)
        for j in range(0, len(sheet_name_xlsx)):
            value[j] = [Cluster.different_item(node_matrixs_difference_R[i][j], item_union[j])]
        book_name_xlsx = './excel/Different_Item/' + node_name[i] + '_right.xlsx'
        TextProcess.write_excel_xlsx(book_name_xlsx, sheet_name_xlsx, value)






func_dict = {'pre_g': preprocess_general,
             'pre_h': preprocess_hull,
             'w_d_e': write_difference_to_excel,
             'cos_den': cos_distance_dendrogram,
             'num_den': num_difference_dendrogram,
             'c&d': common_and_different_items}



def main(run):
    """
    :param run: a list consist of functions to run.
    """
    for func in run:
        func_dict.get(func)()


if __name__ == '__main__':
    run = ['w_d_e']
    main(run)







