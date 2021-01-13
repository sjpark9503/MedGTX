import pandas as pd
import re, json
import argparse

import scispacy, spacy


def config():

    parser = argparse.ArgumentParser()

    parser.add_argument('--load_file_path', type=str, default='p_sections_discharge_summary.csv')
    parser.add_argument('--save_file_path1', type=str, default='p_sections.txt')
    parser.add_argument('--save_file_path2', type=str, default='p_hadm_ids.txt')

    opt = parser.parse_args()

    return opt


def load_csv_file(file_path):
    df = pd.read_csv(file_path)
    return df


def save_txt_file(txt_file, file_path):
    with open(file_path, "w") as file:
        for txt in txt_file:
            file.write(txt + '\n')
            file.write('\n')
    return print('save successfully!')


def preprocess_scispacy(nlp, section_text):
    section_text_p = ' '.join([token.text for token in nlp(section_text)])
    return section_text_p


def main(opt):
    data       = load_csv_file(file_path=opt.load_file_path)
    # data = data.iloc[:10]
    
    # not na 
    data = data[data.TEXT.notna()]

    # length > 200
    data = data[data.TEXT.apply(lambda x: len(x) > 200)]

    # delete ""
    data1 = data.copy()
    data1.TEXT = data.TEXT.apply(lambda x: x[1:-1])

    # preprocessed by scispacy
    nlp = spacy.load("en_core_sci_sm")
    data.TEXT = data1.TEXT.apply(lambda x: preprocess_scispacy(nlp, x))
    del data1
    print('preprocess successfully!')
    

    # recover and extract full info of data(subject_id, hamd_id)
    noteevents = load_csv_file(file_path='file/noteevents.csv')
    noteevents = noteevents[['ROW_ID', 'SUBJECT_ID', 'HADM_ID']]

    # data=p / noteevents
    data1 = noteevents[noteevents.ROW_ID.isin(data.ROW_ID)]
    
    # save txt file
    print('data len: {}', len(data))
    save_txt_file(txt_file=data.TEXT, file_path=opt.save_file_path1)
    hadm_id = data1.HADM_ID.astype(int).astype(str)
    print('data1 len: {}', len(hadm_id))
    save_txt_file(txt_file=hadm_id, file_path=opt.save_file_path2)


if __name__ == '__main__':
    opt = config()
    main(opt=opt)