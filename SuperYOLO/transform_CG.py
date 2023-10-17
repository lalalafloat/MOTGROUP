import os

# 要获取文件名的文件夹路径
folder_path = "/media/ok/4t1/Data/CG"

# 使用os.listdir()函数获取文件夹下的所有文件名
# files = os.listdir(folder_path)

# 打印所有文件名
with open('CG_write.txt', 'w') as file:
    # file.write('aaa')
    for files in os.listdir(folder_path):
        path = os.path.join(folder_path, files)
        print(files)
        if not os.path.isdir(path):
            continue
        for file_name in os.listdir(path):
            print(file_name)
            file.writelines('/media/ok/4t1/Data/CG/' + files + '/' + file_name[:-4] + "\n")