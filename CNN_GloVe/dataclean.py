import csv
import jieba

FILENAME = 'dataset\long_comments_delete_english.csv'
STOPWORD = 'dataset\hit_stopwords.txt'

if __name__ == '__main__':
    # Reading stopwords
    with open(STOPWORD, 'r', encoding='utf-8') as f:
        stopwords = f.readlines()
    for i in range(len(stopwords)):
        stopwords[i] = stopwords[i].rstrip('\n')
    stopwords.append('…')
    # Reading database
    rating, content = [], []
    with open(FILENAME, 'r', encoding='utf-8') as f:
        data = csv.reader(f)
        t = 0
        for row in data:
            t += 1
            if t == 1: continue
            if len(row[6]) > 0 and int(row[6]) == 3:
                continue
            if len(row[3]) != 0 and len(row[6]) != 0:
                rating.append('1' if int(row[6]) > 3 else '0')
                content.append(row[3])
    print('Total lines:', len(rating))
    # jieba processing
    cleaned = []
    for i in range(len(content)):
        tmp = rating[i]
        seglist = jieba.cut(content[i])
        for word in seglist:
            if word not in stopwords:
                tmp += ' '
                tmp += word
        cleaned.append(tmp)
        if i % 10000 == 0: print('Processed', i, 'lines...')
    # Save as file
    print('Writing into file...')
    with open('CNN_GloVe\comments_cleaned.txt', 'w', encoding='utf-8') as f:
        for line in cleaned:
            f.write(line + '\n')