import csv

f = open('comments.csv', 'r', encoding='utf-8')
lf = open('long_comments.csv', 'w', newline="", encoding='utf-8')
tmplf = []
with f:
    reader = csv.reader(f)
    t = 0
    for row in reader:
        t += 1
        if t == 1:
            tmplf.append(row)
            continue
        if len(row[3]) > 10 and len(row[6]) > 0:
            tmplf.append(row)
with lf:
    writer = csv.writer(lf)
    writer.writerows(tmplf)
