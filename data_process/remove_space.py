import os


def rename_directories(root_dir):
    """
    递归遍历并重命名目录中的空格为下划线。

    :param root_dir: 根目录路径
    """
    # 获取当前目录下的所有文件和子目录
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        for dirname in dirnames:
            old_path = os.path.join(dirpath, dirname)
            new_dirname = dirname.replace('-', '_')
            new_path = os.path.join(dirpath, new_dirname)

            # 如果新旧名称不同，则进行重命名
            if old_path != new_path:
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} -> {new_path}")
                except Exception as e:
                    print(f"Failed to rename {old_path}: {e}")


if __name__ == "__main__":
    # 指定要处理的根目录
    root_directory = "/data/users/lph/datasets/USTC_TFC2016"

    if not os.path.isdir(root_directory):
        print(f"错误：{root_directory} 不是一个有效的目录。")
    else:
        print(f"开始处理目录：{root_directory}")
        rename_directories(root_directory)
        print("处理完成！")