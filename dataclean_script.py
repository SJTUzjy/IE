import csv

f = open('comments.csv', 'r', encoding='ISO-8859-1')
lf = open('long_comments.csv', 'w', encoding='ISO-8859-1')
tmplf = []
with f:
    reader = csv.reader(f)
    t = 0
    for row in reader:
        t += 1
        if t == 1:
            tmplf.append(row)
            continue
        if len(row[3]) > 10:
            tmplf.append(row)
with lf:
    writer = csv.writer(lf)
    writer.writerows(tmplf)