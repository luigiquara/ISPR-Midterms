file = open('lercio_dataset.txt', 'r')
lines = file.readlines()

new_lines = []
change = False
for line in lines:
    if line[-2] == ',':
        change = True
        l = list(line)
        l[-2] = ''
        line = ''.join(l)

    new_lines.append(line)

file.close()

print(change)
if change:
    file = open('temp_lercio.txt', 'w')
    file.writelines(new_lines)
    file.close()
