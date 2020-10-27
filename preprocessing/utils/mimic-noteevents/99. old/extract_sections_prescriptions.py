import pandas as pd
import re, json


def load_csv_file(file_path):
    df = pd.read_csv(file_path)
    return df


def save_csv_file(csv_data, file_path):
    csv_data.to_csv(file_path, index=False)
    return print('save successfully!')


def extract_px_section(text):
    px_section = []
    text = json.loads(text) # change string format to dict
    headers, sections = text[0], text[1]
    
    pos1, pos2, pos3, pos4 = -999, -999, -999, -999
    
    h1 = 'discharge medications'
    h2 = 'discharge disposition'
    h3 = 'discharge diagnosis'
    h4 = 'discharge condition'
    
    if h1 in headers:
        pos1 = headers.index(h1)
    if h2 in headers:
        pos2 = headers.index(h2)
    if h3 in headers:
        pos3 = headers.index(h3)
    if h4 in headers:
        pos4 = headers.index(h4)

    if pos1 + pos2 + pos3 + pos4 > 0: # have all together
        if pos1 < pos2 < pos3 < pos4: # well organized
#             px_headers = headers[pos1:pos2]
            px_section = ' '.join(sections[pos1:pos2])
            
    return px_section


def main():
    LOAD_FILE_PATH  = 'preprocessed.csv'
    SAVE_FILE_PATH  = 'preprocessed_prescriptions.csv'
    
    data       = load_csv_file(file_path=LOAD_FILE_PATH)
    notes      = data.TEXT.apply(lambda x: json.dumps(extract_px_section(x)))
    print('extract px section from notes successfully!')
    new_data   = pd.concat([data.ROW_ID, notes], axis=1)
    
    save_csv_file(csv_data=new_data, file_path=SAVE_FILE_PATH)

    
if __name__ == '__main__':
    main()