import pandas as pd
import re, json

import scispacy, spacy


def load_csv_file(file_path):
    df = pd.read_csv(file_path)
    return df


def save_txt_file(txt_file, file_path):
    with open(file_path, "w") as file:
        for txt in txt_file:
            file.write(txt + '\n')
            file.write('\n')
    return print('save successfully!')


def preprocess_prescription_notes(df):
    # notna
    notes = df[df.TEXT.notna()].TEXT
    # length > 200
    notes = notes[notes.apply(lambda x: len(x) > 200)]
    # delete ""
    notes = notes.apply(lambda x: x[1:-1])
    return notes


def preprocess_scispacy(nlp, section_text):
    section_text_p = ' '.join([token.text for token in nlp(section_text)])
    return section_text_p


def main():
    LOAD_FILE_PATH  = 'preprocessed_prescriptions.csv'
    SAVE_FILE_PATH  = 'preprocessed_prescriptions_final.txt'
    
    data       = load_csv_file(file_path=LOAD_FILE_PATH)
    
    # preprocessed by note length
    notes = preprocess_prescription_notes(data)
    
    # preprocessed by scispacy
    nlp = spacy.load("en_core_sci_sm")
    notes = notes.apply(lambda x: preprocess_scispacy(nlp, x))
    print('preprocess successfully!')
    
    # save txt file
    save_txt_file(txt_file=notes, file_path=SAVE_FILE_PATH)

    
if __name__ == '__main__':
    main()
