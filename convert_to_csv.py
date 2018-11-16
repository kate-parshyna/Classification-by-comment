import pandas as pd
import csv

def convert(labels, path, img):
    df = pd.read_csv(path, encoding='windows-1256')
    lenght = len(df['image_name'])
    for i in range(0, lenght):
        if df.iloc[i,df.columns.get_loc('image_name')] == img:
            answer = []
            for label in labels:
                if label in df.iloc[i,df.columns.get_loc('comment')]:
                    answer.append(str(label)+':1')
                else:
                    answer.append(str(label) + ':0')
                df.iloc[i, df.columns.get_loc('is_label')] = " ".join(str(x) for x in answer)
                df.to_csv(path, encoding='utf-8', index=False)



