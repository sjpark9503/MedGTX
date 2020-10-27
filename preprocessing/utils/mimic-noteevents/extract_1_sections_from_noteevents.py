import pandas as pd
import re, json 
import argparse


'''
preprocessing for mimic discharge summary note

1. load NOTEEVENTS.csv

2. get discharge sumamry notes
    a) NOTEVENTS.CATEGORY = 'Discharge Summary'
    b) NOTEVENTS.DESCRIPTION = 'Report'
    c) eliminate a short-note

3. preprocess discharge sumamry notes
    a) clean text
    b) split sections by headers
    
4. save csv file
    a) PK: NOTEVENTS.ROW_ID
    b) TEXT: string(doubled-list)
    
'''


def config():

    parser = argparse.ArgumentParser()

    parser.add_argument('--load_file_path', type=str, default='file/NOTEEVENTS.csv')
    parser.add_argument('--save_file_path', type=str, default='sections_discharge_summary.csv')

    opt = parser.parse_args()

    return opt


def load_noteevents(file_path):

    df = pd.read_csv(file_path)

    # dataframe dtype config
    df.CHARTDATE = pd.to_datetime(df.CHARTDATE, format='%Y-%m-%d', errors='raise')
    df.CHARTTIME = pd.to_datetime(df.CHARTTIME, format='%Y-%m-%d %H:%M:%S', errors='raise')
    df.STORETIME = pd.to_datetime(df.STORETIME)

    return df


def save_csv_file(csv_data, file_path):
    csv_data.to_csv(file_path, index=False)
    print('save successfully!')


def get_discharge_summary(df_notevents):

    cond1 = (df_notevents.CATEGORY == 'Discharge summary')
    cond2 = (df_notevents.DESCRIPTION == 'Report')

    df_discharge_smmary = df_notevents[cond1&cond2]
    df_discharge_smmary = df_discharge_smmary[['ROW_ID', 'TEXT']]
    
    # eliminate a short-note (subject_id=30561, hadm_id=178941)
    df_discharge_smmary = df_discharge_smmary[df_discharge_smmary.TEXT.apply(lambda x: len(x) > 100)]

    return df_discharge_smmary


def pattern_repl(matchobj):
    # Return a replacement string to be used for match object
    return ' '.rjust(len(matchobj.group(0)))  


def clean_text(text):
    # 1. Replace [**Patterns**] with spaces.
    text = re.sub(r'\[\*\*.*?\*\*\]', pattern_repl, text)
    
    # 2. Replace `_` with spaces.
    new_text = re.sub(r'_', ' ', text)
    
    return new_text


def split_section(text):
    headers, sections = [], []
#     pattern = "^([A-z0-9 ]+)(:)|Discharge Date:|Sex:|JOB#:|Unit No:|FOLLOW-UP PLANS:"
    except_pattern = "(?!(Sig:)|(disp:))"
    include_keywords = "(Discharge Date:)|(Sex:)|(JOB#:)|(Unit No:)|(FOLLOW-UP PLANS:)"
    pattern = "^" + except_pattern + "([A-z0-9 ]+)(:)|" + include_keywords
    SEPERATORS = re.compile(pattern, re.I | re.M)
    start = 0
    
    for matcher in SEPERATORS.finditer(text):
        # cut off by the position of later SEPERATOR
        end = matcher.start()
        if end != start: # except for first line
            section = text[start:end]
            if ':' not in section: #
                pass
            else:
                section = section[len(header):].strip() # except for header in section
                sections.append(section)
        start = end
        end = matcher.end()
        
        # collect each title in the beginning of section
        header = text[start:end].lower()
        headers.append(header)
        
    # add last section
    section = text[start:]
    section = section[len(header):].strip()
    sections.append(section)
    
    return headers, sections


def clean_header(header):
    # delete : (colon)
    header = re.sub(r',', '', header)
    new_header = re.sub(r':', '', header)
    new_header = new_header.strip()
    return new_header


def clean_section(section):
    # Replace multiple spaces with a space.
    new_section = ' '.join(section.split())
    return new_section


def preprocess_discharge_summary(text):
    text = clean_text(text)
    headers, sections = split_section(text)
    
    # for duplicated keys problem when formulate dict type data
#     for idx in range(len(headers)):
#         h = clean_header(headers[idx])
#         s = clean_section(sections[idx])
#         result[h] = s
    
    new_headers, new_sections = [], []
    for idx in range(len(headers)):
        h = clean_header(headers[idx])
        s = clean_section(sections[idx])
        new_headers.append(h)
        new_sections.append(s)
    return [new_headers, new_sections]


def main(opt):
    
    data = load_noteevents(file_path=opt.load_file_path)
    print('Load NOTEEVENTS successfully!')
    data       = get_discharge_summary(data)
    print('Get discharge summary successfully!')
    notes      = data.TEXT.apply(lambda x: json.dumps(preprocess_discharge_summary(x)))
    print('Preprocess notes successfully!')
    new_data   = pd.concat([data.ROW_ID, notes], axis=1)

    save_csv_file(csv_data=new_data, file_path=opt.save_file_path)

    
if __name__ == '__main__':
    
    opt = config()
    main(opt=opt)