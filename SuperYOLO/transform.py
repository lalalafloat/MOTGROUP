import os

# 要获取文件名的文件夹路径
folder_path = "/media/ok/4t1/Data/CG/train/images"

# 使用os.listdir()函数获取文件夹下的所有文件名
file_names = os.listdir(folder_path)

# 打印所有文件名
with open('CGtrain_write.txt', 'w') as file:
#     file.write('aaa')
    for file_name in file_names:
        print(file_name)
        file.writelines('/media/ok/4t1/Data/CG/train/images/' + file_name[:-4] + "\n")