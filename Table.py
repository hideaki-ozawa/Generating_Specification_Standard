import os
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import TextProcess
import tabula

def read_table(pdf_path,
               pages='7-10',
               output_dir='./pdf/table'):
    '''

    :param pdf_path:
    :param pages:str  e.g '7-10', 'all'
    :param output_dir:
    :return:
    '''

    os.makedirs(output_dir, exist_ok=True)

    print("pdf: {0}  pages: {1}".format(pdf_path[-14:], pages))

    try:
        ln = tabula.read_pdf(pdf_path, lattice=True, pages=pages)
        page_ln = range(len(ln))
        txt = ''
        for i, page in enumerate(page_ln):
            df = ln[i]

            if df.empty:
                pass
            else:
                # print(df)
                page_str = column2str(df)
                for index, row in df.iterrows():
                    row_str = ' '.join(map(str, row.dropna()))
                    row_str = row_str.replace('\r', ' ')
                    # print('------------\nrow: {0}'.format(row_str))
                    page_str += row_str + '\n\n'

                txt += page_str
        output_name = output_dir + '/' + pdf_path[-14:-4] + "_page" + pages + ".txt"
        with open(output_name, 'w') as f:
            f.write(txt)
        print(output_name + 'Writing Succeed!')

    except Exception as e:
        print(e)


def column2str(df):
    result = ''
    for col in df.columns:
        if 'Unnamed' in col:
            pass
        else:
            result += col.replace('\r', ' ') + ' '
    result += '\n\n'
    return result


def table2txt(input_path, subsection, title, head, tail):
    '''
    split .txt file(full text) into each section and subsection.

    :param input_path: txt file path
    :param subsection: m×n dim. m=section, n=subsection
    :param save_path: dir. e.g. "./txt/Machinery_Part"
    :return:
    '''
    print("txt: {0}  Section:{1}".format(input_path, subsection))

    fp = open(input_path)
    data = fp.read()

    result = title + '\n\n'

    dir = './table/Machinery_Part/Section'
    section = TextProcess.search_num(subsection)[0]
    shipname = input_path[-22:-12]
    save_path = dir + section + '/' + subsection + '/'
    os.makedirs(save_path, exist_ok=True)

    # num of space is different accoding to how to convert pdf to txt
    start_id = data.find(head)

    end_id = data.rfind(tail)
    end_id = [m.end() for m in re.finditer(tail, data)][-1] + 1
    if end_id != -1:
        result += data[start_id:end_id]

    result = result.rstrip()
    wf = open(save_path + shipname + '.txt', "w")
    wf.write(result)
    wf.close()
    print(save_path + ' Writing Succeed!')


if __name__ == '__main__':
    txt_path = TextProcess.get_txt_path('./pdf/table')
    for txt in txt_path:
        table2txt(input_path=txt,
                  subsection='_1_2',
                  title='1.2  MACHINERY PARTICULAR.',
                  head='Rule ',
                  tail=' In Double Bottom')
        table2txt(input_path=txt,
                  subsection='_6_3',
                  title='1.2  MACHINERY PARTICULAR.',
                  head='Rule ',
                  tail=' In Double Bottom')
