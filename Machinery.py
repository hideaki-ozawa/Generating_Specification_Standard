import TextProcess
from TextProcess import subsection_value
import Cluster
import Table

import shutil
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

abbreviation = ['reg', 'i.e', 'no', 'incl', 'vessel', 'res', 'rev', 'msc',
                'f.o', 'pty', 'ltd', 'code', 'inc', 'abt', 'i.c.t.m', 'm.c.r',
                'dwg', 'w.b.t', 'nos', 'm.t', 'Ext', 'G.T.D.W', 'No']
txt_path = './txt/Machinery_Part/section'
pdf_path = TextProcess.get_pdf_path('./pdf/machinery_pdf')
shipname = [None]*len(pdf_path)

for i in range(0, len(pdf_path)):
    shipname[i] = pdf_path[i][-13:-9]


def preprocess_general():
    """
    Put the pdf file in './pdf/input'.
    Get corresponding txt file(general part) in './pdf/output1'.
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
    Put the pdf file in './pdf/input'.
    Get corresponding txt file(hull part) in './pdf/output1'.
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


def preprocess_machinery():
    """
    Put the pdf file in './pdf/input'.
    Get corresponding txt file(general part) in './pdf/output1'.
    """
    shutil.rmtree('./pdf/output1')
    os.mkdir('./pdf/output1')

    pdf_path = TextProcess.get_pdf_path('./pdf/machinery_pdf')

    for i in range(0, len(pdf_path)):
        TextProcess.split_pdf_endpage(6, pdf_path[i], './pdf/output0/' + pdf_path[i][-14:])
    pdf_path = TextProcess.get_pdf_path('./pdf/output0')
    for i in range(0, len(pdf_path)):
        save_name = './pdf/output1/' + pdf_path[i][-14:-4] + '.txt'
        # TextProcess.convert_pdf_to_txt(pdf_path[i], save_name)
        TextProcess.convert_pdf_to_txt_by_tika(pdf_path[i], save_name)
    txt_path = TextProcess.get_txt_path('./pdf/output1')
    for i in range(0, len(txt_path)):
        TextProcess.remove_specific_char(txt_path[i])
        # TextProcess.remove_n(txt_path[i])

    shutil.rmtree('./pdf/output0')
    os.mkdir('./pdf/output0')
    # shutil.rmtree('./pdf/input')
    # os.mkdir('./pdf/input')


def split_txt():

    sec_1 = ['1.1', '1.2', '1.3', '1.4']
    sec_2 = ['2.1', '2.2']  # 2.2&2.3
    sec_3 = ['3.1', '3.2', '3.3', '3.4', '3.5', '3.6']
    sec_4 = ['4.1', '4.2']
    sec_5 = ['5.1', '5.2', '5.3']
    sec_6 = ['6.1', '6.2', '6.3', '6.4']    # '6.4-6.7'
    sec_7 = ['7.1', '7.2', '7.3', '7.4']    # 7.4&7.5
    sec_8 = ['8.1', '8.2']
    sec_9 = ['9.1', '9.2']
    sec_10 = ['10.1', '10.2', '10.3', '10.4', '10.5', '10.6', '10.7']
    sec_11 = ['11.1', '11.2']
    sec_12 = ['12.1', '12.2', '12.3', '12.4', '12.10']   # 12.4-12.9
    sec_13 = ['13.1', '13.2', '13.3', '13.4', '13.5']
    sec_14 = ['14.1', '14.2', '14.3', '14.4', '14.5', '14.6', '14.7', '14.8',
              '14.9']
    sec_15 = ['15.1', '15.2', '15.3', '15.4', '15.5', '15.6', '15.7', '15.8']

    sections = [sec_1, sec_2, sec_3, sec_4, sec_5, sec_6, sec_7, sec_8, sec_9,
                sec_10, sec_11, sec_12, sec_13, sec_14, sec_15]

    save_path = "./txt/Machinery_Part"
    txt_path = TextProcess.get_txt_path('./pdf/preprocess_output1')
    shutil.rmtree(save_path)
    for txt in txt_path:
        TextProcess.split_txt(txt, sections, save_path)

    txt_path = TextProcess.get_txt_path('./txt/Machinery_Part')
    for txt in txt_path:
        TextProcess.remove_illegal_char(txt)
        TextProcess.add_dot(txt)


def write_difference_to_excel():
    """
    Find the difference between specifications and union set
    and write it into xlsx file.
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


def write_difference_to_excel_machinery():
    """
    Find the difference between specifications and union set
    and write it into xlsx file.
    """

    os.makedirs('./excel/Machinery_Part', exist_ok=True)

    subsec1 = ['_1_1', '_1_3', '_1_4']  # '_1_2'
    subsec2 = ['_2_1', '_2_2']  # 2.2&2.3
    subsec3 = ['_3_1', '_3_2', '_3_3', '_3_4', '_3_5', '_3_6']
    subsec4 = ['_4_1', '_4_2']
    subsec5 = ['_5_1', '_5_2', '_5_3']
    subsec6 = ['_6_1', '_6_2', '_6_3', '_6_4']  # '6.4&6.5&6.6&6.7'
    subsec7 = ['_7_1', '_7_2', '_7_3', '_7_4']  # 7.4&7.5
    subsec8 = ['_8_2']  # '_8_1'
    subsec9 = ['_9_1', '_9_2']
    subsec10 = ['_10_1', '_10_2', '_10_3']  # '_10_4' '_10_5' '_10_6' '_10_7'
    subsec11 = ['_11_1', '_11_2']
    subsec12 = ['_12_1', '_12_2', '_12_3', '_12_4', '_12_10']  # '_12_5'
    subsec13 = ['_13_1', '_13_2', '_13_3']  # '_13_4' '_13_5'
    subsec14 = ['_14_1', '_14_2', '_14_3', '_14_4', '_14_5', '_14_6', '_14_7',
                '_14_8', '_14_9']
    subsec15 = ['_15_1', '_15_2', '_15_3', '_15_4', '_15_5', '_15_6', '_15_7',
                '_15_8']

    subsections = [subsec1, subsec2, subsec3, subsec4, subsec5,
                   subsec6, subsec7, subsec8, subsec9, subsec10,
                   subsec11, subsec12, subsec13, subsec14, subsec15]

    TextProcess.subsection_value(shipname, subsections, abbreviation, txt_path)



def cos_distance_dendrogram():
    """
    Calculate the cosine distance between any 2 of the specifications
    and write the matrix into xlsx file.
    Draw the dendrogram with the matrix.
    """
    subsection = ['_1_1', '_1_3', '_1_4']  # '_1_2'
    matrix_1 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_2_1', '_2_2', '_2_3']
    matrix_2 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_3_1', '_3_2', '_3_3', '_3_4', '_3_5', '_3_6']
    matrix_3 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_4_1', '_4_2']
    matrix_4 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_5_1', '_5_2', '_5_3']
    matrix_5 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_6_1', '_6_4', '_6_5', '_6_6', '_6_7']  # '_6_2' '_6_3'
    matrix_6 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_7_1', '_7_2', '_7_3', '_7_4', '_7_5']
    matrix_7 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_8_2']  # '_8_1'
    matrix_8 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_9_1', '_9_2']
    matrix_9 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_10_1', '_10_2', '_10_3']  # '_10_4' '_10_5' '_10_6' '_10_7'
    matrix_10 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_11_1', '_11_2']
    matrix_11 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_12_1', '_12_2', '_12_3', '_12_4', '_12_6', '_12_7', '_12_8', '_12_9', '_12_10']  # '_12_5'
    matrix_12 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_13_1', '_13_2', '_13_3']  # '_13_4' '_13_5'
    matrix_13 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_14_1', '_14_2', '_14_3', '_14_4', '_14_5', '_14_6', '_14_7', '_14_8', '_14_9']
    matrix_14 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_15_1', '_15_2', '_15_3', '_15_4', '_15_5', '_15_6', '_15_7', '_15_8']
    matrix_15 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)

    matrix = []
    for i in range(0, len(shipname)):
        matrix.append(
            matrix_1[i] + matrix_2[i] + matrix_3[i] + matrix_4[i] +
            matrix_5[i] + matrix_6[i] + matrix_7[i] + matrix_8[i] +
            matrix_9[i] + matrix_10[i] + matrix_11[i] + matrix_12[i] +
            matrix_13[i] + matrix_14[i] + matrix_15[i])

    matrix = np.array(matrix)
    distance_matrix = TextProcess.cos_distance(matrix)

    rows = len(shipname) + 1
    cols = len(shipname) + 1
    value_distance = [[0] * cols for i in range(rows)]

    fullname = [None]*len(shipname)
    for i in range(0, len(shipname)):
        fullname[i] = 'S' + shipname[i] + 'M9000'

    value_distance[0][0] = ''
    for i in range(1, cols):
        value_distance[0][i] = fullname[i - 1]
    for i in range(1, rows):
        value_distance[i][0] = fullname[i - 1]
    for i in range(1, rows):
        for j in range(1, cols):
            value_distance[i][j] = distance_matrix[i - 1][j - 1]

    book_name_xlsx = './excel/Cosine_Similarity_Machinery.xlsx'
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
                   matrix_4,
                   matrix_5,
                   matrix_6,
                   matrix_7,
                   matrix_8,
                   matrix_9,
                   matrix_10,
                   matrix_11,
                   matrix_12,
                   matrix_13,
                   matrix_14,
                   matrix_15
                   ]
    section_name = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']


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
    subsection = ['_1_1', '_1_3', '_1_4']  # '_1_2'
    matrix_1 = TextProcess.section_matrix(shipname, subsection, abbreviation,
                                          txt_path)
    subsection = ['_2_1', '_2_2', '_2_3']
    matrix_2 = TextProcess.section_matrix(shipname, subsection, abbreviation,
                                          txt_path)
    subsection = ['_3_1', '_3_2', '_3_3', '_3_4', '_3_5', '_3_6']
    matrix_3 = TextProcess.section_matrix(shipname, subsection, abbreviation,
                                          txt_path)
    subsection = ['_4_1', '_4_2']
    matrix_4 = TextProcess.section_matrix(shipname, subsection, abbreviation,
                                          txt_path)
    subsection = ['_5_1', '_5_2', '_5_3']
    matrix_5 = TextProcess.section_matrix(shipname, subsection, abbreviation,
                                          txt_path)
    subsection = ['_6_1', '_6_4', '_6_5', '_6_6', '_6_7']  # '_6_2' '_6_3'
    matrix_6 = TextProcess.section_matrix(shipname, subsection, abbreviation,
                                          txt_path)
    subsection = ['_7_1', '_7_2', '_7_3', '_7_4', '_7_5']
    matrix_7 = TextProcess.section_matrix(shipname, subsection, abbreviation,
                                          txt_path)
    subsection = ['_8_2']  # '_8_1'
    matrix_8 = TextProcess.section_matrix(shipname, subsection, abbreviation,
                                          txt_path)
    subsection = ['_9_1', '_9_2']
    matrix_9 = TextProcess.section_matrix(shipname, subsection, abbreviation,
                                          txt_path)
    subsection = ['_10_1', '_10_2', '_10_3']  # '_10_4' '_10_5' '_10_6' '_10_7'
    matrix_10 = TextProcess.section_matrix(shipname, subsection, abbreviation,
                                           txt_path)
    subsection = ['_11_1', '_11_2']
    matrix_11 = TextProcess.section_matrix(shipname, subsection, abbreviation,
                                           txt_path)
    subsection = ['_12_1', '_12_2', '_12_3', '_12_4', '_12_6', '_12_7',
                  '_12_8', '_12_9', '_12_10']  # '_12_5'
    matrix_12 = TextProcess.section_matrix(shipname, subsection, abbreviation,
                                           txt_path)
    subsection = ['_13_1', '_13_2', '_13_3']  # '_13_4' '_13_5'
    matrix_13 = TextProcess.section_matrix(shipname, subsection, abbreviation,
                                           txt_path)
    subsection = ['_14_1', '_14_2', '_14_3', '_14_4', '_14_5', '_14_6',
                  '_14_7', '_14_8', '_14_9']
    matrix_14 = TextProcess.section_matrix(shipname, subsection, abbreviation,
                                           txt_path)
    subsection = ['_15_1', '_15_2', '_15_3', '_15_4', '_15_5', '_15_6',
                  '_15_7', '_15_8']
    matrix_15 = TextProcess.section_matrix(shipname, subsection, abbreviation,
                                           txt_path)

    matrix = []
    for i in range(0, len(shipname)):
        matrix.append(
            matrix_1[i] + matrix_2[i] + matrix_3[i] + matrix_4[i] +
            matrix_5[i] + matrix_6[i] + matrix_7[i] + matrix_8[i] +
            matrix_9[i] + matrix_10[i] + matrix_11[i] + matrix_12[i] +
            matrix_13[i] + matrix_14[i] + matrix_15[i])

    matrix = np.array(matrix)
    difference = TextProcess.num_difference_matrix(matrix)

    rows = len(shipname) + 1
    cols = len(shipname) + 1
    value_distance = [[0] * cols for i in range(rows)]

    fullname = [None] * len(shipname)
    for i in range(0, len(shipname)):
        fullname[i] = 'S' + shipname[i] + 'M9000'

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


def labeling_each_section():
    subsection = ['_1_1', '_1_3', '_1_4']  # '_1_2'
    matrix_1 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_2_1', '_2_2', '_2_3']
    matrix_2 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_3_1', '_3_2', '_3_3', '_3_4', '_3_5', '_3_6']
    matrix_3 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_4_1', '_4_2']
    matrix_4 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_5_1', '_5_2', '_5_3']
    matrix_5 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_6_1', '_6_4', '_6_5', '_6_6', '_6_7']  # '_6_2' '_6_3'
    matrix_6 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_7_1', '_7_2', '_7_3', '_7_4', '_7_5']
    matrix_7 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_8_2']  # '_8_1'
    matrix_8 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_9_1', '_9_2']
    matrix_9 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_10_1', '_10_2', '_10_3']  # '_10_4' '_10_5' '_10_6' '_10_7'
    matrix_10 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_11_1', '_11_2']
    matrix_11 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_12_1', '_12_2', '_12_3', '_12_4', '_12_6', '_12_7', '_12_8', '_12_9', '_12_10']  # '_12_5'
    matrix_12 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_13_1', '_13_2', '_13_3']  # '_13_4' '_13_5'
    matrix_13 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_14_1', '_14_2', '_14_3', '_14_4', '_14_5', '_14_6', '_14_7', '_14_8', '_14_9']
    matrix_14 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)
    subsection = ['_15_1', '_15_2', '_15_3', '_15_4', '_15_5', '_15_6', '_15_7', '_15_8']
    matrix_15 = TextProcess.section_matrix(shipname, subsection, abbreviation, txt_path)


    matrix_list = [matrix_1,
                   matrix_2,
                   matrix_3,
                   matrix_4,
                   matrix_5,
                   matrix_6,
                   matrix_7,
                   matrix_8,
                   matrix_9,
                   matrix_10,
                   matrix_11,
                   matrix_12,
                   matrix_13,
                   matrix_14,
                   matrix_15
                   ]
    section_name = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                    '11', '12', '13', '14', '15']

    # make value distance for labeling xlsx
    rows = len(shipname) + 1
    cols = len(section_name) + 1
    print("rows: {0}, cols: {1}".format(rows, cols))
    value_distance = [[0] * cols for i in range(rows)]

    value_distance[0][0] = ''
    for i in range(1, cols):
        value_distance[0][i] = 'SECTION' + section_name[i - 1]
    for i in range(1, rows):
        value_distance[i][0] = shipname[i - 1][:5]

    c_list = []
    dic = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H'}

    # dendrogram in each section
    for i in range(0, len(matrix_list)):
        matrix = matrix_list[i]
        matrix = np.array(matrix)
        distance_matrix = spatial.distance.pdist(matrix, 'cosine')
        Z = linkage(distance_matrix, 'ward')
        t = 0.5 * max(Z[:, 2])
        c = fcluster(Z, t, criterion="distance")
        c_list.append(c)

    book_name_xlsx = './excel/Label_Machinery.xlsx'
    sheet_name_xlsx = 'Label'

    for i in range(1, rows):
        for j in range(1, cols):
            value_distance[i][j] = dic[c_list[j - 1][i - 1]]

    TextProcess.write_matrix_excel_xlsx(book_name_xlsx,
                                        sheet_name_xlsx,
                                        value_distance)


def labeling_each_subsection():
    subsec1 = ['_1_1', '_1_3', '_1_4']  # '_1_2'
    subsec2 = ['_2_1', '_2_2', '_2_3']
    subsec3 = ['_3_1', '_3_2', '_3_3', '_3_4', '_3_5', '_3_6']
    subsec4 = ['_4_1', '_4_2']
    subsec5 = ['_5_1', '_5_2', '_5_3']
    subsec6 = ['_6_1', '_6_4', '_6_5', '_6_6', '_6_7']  # '_6_2' '_6_3'
    subsec7 = ['_7_1', '_7_2', '_7_3', '_7_4', '_7_5']
    subsec8 = ['_8_2']  # '_8_1'
    subsec9 = ['_9_1', '_9_2']
    subsec10 = ['_10_1', '_10_2', '_10_3']  # '_10_4' '_10_5' '_10_6' '_10_7'
    subsec11 = ['_11_1', '_11_2']
    subsec12 = ['_12_1', '_12_2', '_12_3', '_12_4', '_12_6', '_12_7', '_12_8', '_12_9', '_12_10']  # '_12_5'
    subsec13 = ['_13_1', '_13_2', '_13_3']  # '_13_4' '_13_5'
    subsec14 = ['_14_1', '_14_2', '_14_3', '_14_4', '_14_5', '_14_6', '_14_7', '_14_8', '_14_9']
    subsec15 = ['_15_1', '_15_2', '_15_3', '_15_4', '_15_5', '_15_6', '_15_7', '_15_8']

    subsections = subsec1 + subsec2 + subsec3 + subsec4 + subsec5 +\
                  subsec6 + subsec7 + subsec8 + subsec9 + subsec10 +\
                  subsec11 + subsec12 + subsec13 + subsec14 + subsec15



    # make value distance for labeling xlsx
    rows = len(shipname) + 1
    cols = len(subsections) + 1
    print("rows: {0}, cols: {1}".format(rows, cols))
    value_distance = [[0] * cols for i in range(rows)]

    value_distance[0][0] = ''
    for i in range(1, cols):
        value_distance[0][i] = subsections[i - 1]
    for i in range(1, rows):
        value_distance[i][0] = shipname[i - 1][:5]

    c_list = []
    alphabet = 'ABCDEFGHIJ'
    dic = {i+1: alphabet[i] for i in range(10)}

    # dendrogram in each section
    for subsec in subsections:
        matrix = TextProcess.subsection_matrix(shipname, subsec, abbreviation, txt_path)
        matrix = np.array(matrix)
        distance_matrix = spatial.distance.pdist(matrix, 'cosine')
        Z = linkage(distance_matrix, 'ward')

        t = 3
        c = fcluster(Z, t, criterion="maxclust")
        c_list.append(c)

    book_name_xlsx = './excel/Label_Machinery.xlsx'
    sheet_name_xlsx = 'Label'

    print("c_list: ", len(c_list))

    for i in range(1, rows):
        for j in range(1, cols):
            value_distance[i][j] = dic[c_list[j - 1][i - 1]]

    TextProcess.write_matrix_excel_xlsx(book_name_xlsx,
                                        sheet_name_xlsx,
                                        value_distance)

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


def read_table():
    pdf_path = TextProcess.get_pdf_path('./pdf/machinery_pdf')
    # pdf_path.remove('./pdf/S0937M9000.pdf')
    output_dir = './pdf/table'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    for pdf in pdf_path:
        Table.read_table(pdf, pages='all')


def subsection_dendrograms():
    """
    Draw the dendrogram of each subsection and make the label list.
    """
    subsec1 = ['_1_1', '_1_3', '_1_4']  # '_1_2'
    subsec2 = ['_2_1', '_2_2']  # 2.2&2.3
    subsec3 = ['_3_1', '_3_2', '_3_3', '_3_4', '_3_5', '_3_6']
    subsec4 = ['_4_1', '_4_2']
    subsec5 = ['_5_1', '_5_2', '_5_3']
    subsec6 = ['_6_1', '_6_2', '_6_3', '_6_4']  # '6.4&6.5&6.6&6.7'
    subsec7 = ['_7_1', '_7_2', '_7_3', '_7_4']  # '7.4&_7_5'
    subsec8 = ['_8_2']  # '_8_1'
    subsec9 = ['_9_1', '_9_2']
    subsec10 = ['_10_1', '_10_2', '_10_3']  # '_10_4' '_10_5' '_10_6' '_10_7'
    subsec11 = ['_11_1', '_11_2']
    subsec12 = ['_12_1', '_12_2', '_12_3', '_12_4', '_12_10']  # 12.5-12.9
    subsec13 = ['_13_1', '_13_2', '_13_3', '_13_4']  # '_13_5'
    subsec14 = ['_14_1', '_14_2', '_14_3', '_14_4', '_14_5', '_14_6', '_14_7',
                '_14_8', '_14_9']
    subsec15 = ['_15_1', '_15_2', '_15_3', '_15_4', '_15_5', '_15_6', '_15_7',
                '_15_8']

    subsections = [subsec1, subsec2, subsec3, subsec4, subsec5,
                   subsec6, subsec7, subsec8, subsec9, subsec10,
                   subsec11, subsec12, subsec13, subsec14, subsec15]

    num_cluster = [2, 3, 5,                         # section1
                   1, 5,                            # section2
                   1, 5, 1, 4, 1, 1,                # section3
                   1, 3,                            # section4
                   1, 4, 2,                         # section5
                   1, 4, 3, 3,                      # section6
                   5, 3, 6, 5,                      # section7
                   2,                               # section8
                   1, 2,                            # section9
                   3, 1, 3,                         # section10
                   5, 4,                            # section11
                   5, 2, 3, 4, 3,                   # section12
                   2, 2, 7, 4,                      # section13
                   1, 5, 2, 3, 1, 3, 3, 2, 2,       # section14
                   1, 3, 2, 2, 2, 1, 1, 6]          # section15

    label_name = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    value = [[''] + shipname]
    m = 0

    for i in range(0, len(subsections)):
        for j in range(0, len(subsections[i])):
            print("Labeling {0}".format(subsections[i][j][1:]), end=' ...')
            matrix = TextProcess.subsection_matrix(shipname, subsections[i][j],
                                                   abbreviation, txt_path)
            matrix = np.array(matrix)
            distance_matrix = spatial.distance.pdist(matrix, 'cosine')
            Z = linkage(distance_matrix, 'ward')

            cluster_label = fcluster(Z, t=num_cluster[m], criterion='maxclust')
            subsec_num = TextProcess.search_num(subsections[i][j])[1]
            alpha = [str(i+1) + '.' + subsec_num]
            for k in range(0, len(cluster_label)):
                n = cluster_label[k] - 1
                alpha.append(label_name[n])
            alpha.append(num_cluster[m])
            value.append(alpha)

            labelList = shipname
            threshold = np.sort(Z[:, 2])[::-1][num_cluster[m] - 2]
            dendrogram(Z, color_threshold=threshold, labels= labelList)
            plt.subplots_adjust(left=0.15, right=0.97, top=0.88, bottom=0.15)
            plt.title('SUBSECTION: {0}\nNUM OF CLUSTER: {1}'.format(subsections[i][j][1:], num_cluster[m]))
            plt.xlabel('SHIP NAME')
            plt.ylabel('DISTANCE')
            fig_path = './dendrogram/chapter_dendrogram/Section' + str(i+1)
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)
            plt.savefig(fig_path + '/' + subsections[i][j] + '.png')
            plt.cla()
            m = m + 1
            print("Suceed.")

    value = [value]
    book_name_xlsx = './excel/label list0.xlsx'
    TextProcess.write_excel_xlsx(book_name_xlsx, ['label_list'], value)


def decide_num_of_cluster():
    subsec1 = ['_1_1', '_1_3', '_1_4']  # '_1_2'
    subsec2 = ['_2_1', '_2_2', '_2_3']
    subsec3 = ['_3_1', '_3_2', '_3_3', '_3_4', '_3_5', '_3_6']
    subsec4 = ['_4_1', '_4_2']
    subsec5 = ['_5_1', '_5_2', '_5_3']
    subsec6 = ['_6_1', '_6_4', '_6_5', '_6_6', '_6_7']  # '_6_2' '_6_3'
    subsec7 = ['_7_1', '_7_2', '_7_3', '_7_4', '_7_5']
    subsec8 = ['_8_2']  # '_8_1'
    subsec9 = ['_9_1', '_9_2']
    subsec10 = ['_10_1', '_10_2', '_10_3']  # '_10_4' '_10_5' '_10_6' '_10_7'
    subsec11 = ['_11_1', '_11_2']
    subsec12 = ['_12_1', '_12_2', '_12_3', '_12_4', '_12_6', '_12_7', '_12_8',
                '_12_9', '_12_10']  # '_12_5'
    subsec13 = ['_13_1', '_13_2', '_13_3']  # '_13_4' '_13_5'
    subsec14 = ['_14_1', '_14_2', '_14_3', '_14_4', '_14_5', '_14_6', '_14_7',
                '_14_8', '_14_9']
    subsec15 = ['_15_1', '_15_2', '_15_3', '_15_4', '_15_5', '_15_6', '_15_7',
                '_15_8']

    subsections = [subsec1, subsec2, subsec3, subsec4, subsec5,
                   subsec6, subsec7, subsec8, subsec9, subsec10,
                   subsec11, subsec12, subsec13, subsec14, subsec15]

    for i in range(0, len(subsections)):
        for j in range(0, len(subsections[i])):
            matrix = TextProcess.subsection_matrix(shipname, subsections[i][j],
                                                   abbreviation, txt_path)
            matrix = np.array(matrix)
            distance_matrix = spatial.distance.pdist(matrix, 'cosine')
            Z = linkage(distance_matrix, 'ward')

            silhouette_coefficient = []
            calinski_harabasz = []
            davies_bouldin = []

            NUM_CLUSTERS_RANGE = range(2, 11)
            for num in NUM_CLUSTERS_RANGE:
                labels = fcluster(Z, t=num, criterion='maxclust')

                # silhouette_coefficient.append(silhouette_score(matrix, labels))
                # calinski_harabasz.append(calinski_harabasz_score(matrix, labels))
                # davies_bouldin.append(davies_bouldin_score(matrix, labels))

            fig = plt.figure()
            fig.subplots_adjust(bottom=0.3, right=0.75)
            host = fig.add_subplot(111)

            par1 = host.twinx()
            par2 = host.twinx()

            labels = ['Silhouette Coefficient',
                      'Calinski Harabasz Index',
                      'Davies Bouldin Index']

            p0, = host.plot(NUM_CLUSTERS_RANGE, silhouette_coefficient, 'bo-',
                            label=labels[0])
            p1, = par1.plot(NUM_CLUSTERS_RANGE, calinski_harabasz, 'rd-',
                            label=labels[1])
            p2, = par2.plot(NUM_CLUSTERS_RANGE, davies_bouldin, 'gs-',
                            label=labels[2])

            host.set_xlabel('Number of Clusters')
            host.set_ylabel(labels[0])
            par1.set_ylabel(labels[1])
            par2.set_ylabel(labels[2])

            par2.spines['right'].set_position(('axes', 1.15))

            lines = [p0, p1, p2]
            host.legend(lines,
                        [l.get_label() for l in lines],
                        fontsize=8,
                        bbox_to_anchor=(0.7, -0.1),
                        loc='upper left')
            fig_path = './fig/num_clusters/Section' + str(i+1)
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)
            plt.savefig(fig_path + '/' + subsections[i][j] + '.png')
            plt.cla()


def machine_learning():
    TextProcess.machine_learning()


func_dict = {'pre_g': preprocess_general,
             'pre_h': preprocess_hull,
             'pre_m': preprocess_machinery,
             'splitxt': split_txt,
             'w_d_e': write_difference_to_excel,
             'w_d': write_difference_to_excel_machinery,
             'cos_den': cos_distance_dendrogram,
             'num_den': num_difference_dendrogram,
             'c&d': common_and_different_items,
             'read_table': read_table,
             'subsec_den': subsection_dendrograms,
             'ml': machine_learning
             }


def main(run):
    """
    :param run: a list consist of functions to run.
    """
    for func in run:
        func_dict.get(func)()


if __name__ == '__main__':
    run = []
    main(run)
