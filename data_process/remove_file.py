import subprocess
import os
import glob

# 文件所在的根目录
root_directory = '/data/users/lph/projects/WTMamba/datasets/ISCX_VPN_2016_APP/pcap'  # 将此路径替换为你的实际路径

# 遍历00到99
for i in range(100):
    file_number = f"{i:02d}"  # 生成三位数的字符串，如 '000', '001', ..., '999'
    file_pattern = f"*UDP*{file_number}.pcap"  # 构造文件名模式

    # 使用os.walk遍历目录树
    for dirpath, dirnames, filenames in os.walk(root_directory):
        # 使用os.path.join构造完整的文件路径模式
        full_pattern = os.path.join(dirpath, file_pattern)

        # 使用glob模块匹配文件名模式
        matching_files = glob.glob(full_pattern)

        if matching_files:
            try:
                # 调用 rm 命令，并使用 -- 来防止文件名中包含破折号被误认为是选项
                result = subprocess.run(['rm', '--'] + matching_files, check=True)
                print(f"Deleted files matching pattern: {full_pattern}")
            except subprocess.CalledProcessError as e:
                print(f"Error executing command for pattern {full_pattern}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
        else:
            print(f"No files found matching pattern in {dirpath}: {full_pattern}")



# file_number = f"UDP"
# file_pattern = f"*{file_number}*"  # 构造文件名模式
#
# # 使用os.walk遍历目录树
# for dirpath, dirnames, filenames in os.walk(root_directory):
#     # 使用os.path.join构造完整的文件路径模式
#     full_pattern = os.path.join(dirpath, file_pattern)
#
#     # 使用glob模块匹配文件名模式
#     matching_files = glob.glob(full_pattern)
#
#     if matching_files:
#         try:
#             # 调用 rm 命令，并使用 -- 来防止文件名中包含破折号被误认为是选项
#             result = subprocess.run(['rm', '--'] + matching_files, check=True)
#             print(f"Deleted files matching pattern: {full_pattern}")
#         except subprocess.CalledProcessError as e:
#             print(f"Error executing command for pattern {full_pattern}: {e}")
#         except Exception as e:
#             print(f"An unexpected error occurred: {e}")
#     else:
#         print(f"No files found matching pattern in {dirpath}: {full_pattern}")