import csv
import re

# test_lf=open('test_long_comments.csv', 'w', newline="", encoding='utf-8')


def rm_english(s):
    return re.sub(r'[^(\u4e00-\u9fa5\u2014\u2018\u2019\u201c\u201d\u2026\u3001\u3002\u300a\u300b\u300e\u300f\u3010\u3011\uff01\uff08\uff09\uff0c\uff1a\uff1b\uff1f)]', '', s)

if __name__ == '__main__':
    tmplf = []
    with open('comments.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        t = 0
        for row in reader:
            t += 1
            if t == 1:
                tmplf.append(row)
                continue
            if len(row[6]) > 0 and int(row[6]) == 3:
                continue
            if len(row[3]) > 10 and len(row[6]) > 0:
                if rm_english(row[3]) == row[3]:
                    tmplf.append(row)
            # Small data to test binary classify.
            if t > 200000: break
    with open('long_comments_delete_english_small.csv', 'w', newline="", encoding='utf-8') as lf:
        writer = csv.writer(lf)
        writer.writerows(tmplf)
