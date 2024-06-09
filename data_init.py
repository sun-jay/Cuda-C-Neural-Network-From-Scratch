# run this after cloning to initialize the data.txt file
# we needed to split it to commit to github

with open('data1.txt', 'r') as file1:
    data1 = file1.readlines()

with open('data2.txt', 'r') as file2:
    data2 = file2.readlines()

with open('data.txt', 'w') as file:
    file.writelines(data1)
    file.writelines(data2)