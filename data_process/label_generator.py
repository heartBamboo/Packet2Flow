import os
import json

def get_file_labels():
    # 获取当前工作目录
    current_directory = "/data/users/lph/projects/WTMamba/datasets/ISCX_VPN_2016_APP/pcap"
    labels = {}

    # 遍历当前目录下的所有子文件夹
    for folder_name in os.listdir(current_directory):
        folder_path = os.path.join(current_directory, folder_name)

        # 检查是否为文件夹
        if os.path.isdir(folder_path):
            # 初始化子文件夹的文件列表
            labels[folder_name] = []

            # 遍历子文件夹中的所有文件
            for file_name in os.listdir(folder_path):
                # 只添加以 .pcap 结尾的文件
                if file_name.endswith('.pcap'):
                    labels[folder_name].append(file_name)

    return labels


def save_labels_to_json(labels):
    # 将 labels 字典写入 labels.json 文件
    with open('/data/users/lph/projects/WTMamba/datasets/ISCX_VPN_2016_APP/raw/labels.json', 'a') as json_file:
        json.dump(labels, json_file, indent=4)


if __name__ == "__main__":
    labels = get_file_labels()
    save_labels_to_json(labels)
    print("Labels have been written to labels.json")