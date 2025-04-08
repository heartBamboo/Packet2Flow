import argparse
import os
import dpkt
from dpkt.pcap import DLT_RAW, DLT_EN10MB, DLT_IEEE802_11, DLT_LINUX_SLL
import shutil
import glob
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 定义全局锁以确保线程安全
lock = threading.Lock()

interval_seconds = {
    'ISCX_VPN_2016': 0.128,
    'ISCX_Tor_2017': 0.128,
    'ISCXTor2016': 0.128,
    'CIC_IOT_2022': 0.512,
    'USTC_TFC2016': 0.128,
    'ISCX_Bot_2014': 0.128,
    'CIC_IDS_2017': 0.128,
    'VNAT': 0.128,
    'CSTNET_TLS_1.3': 0.128


}

def split_pcap_by_time(input_file, output_dir, time_interval, args):
    flow_end_ts = -1
    flows = []  # [flow, ...]
    flow = []  # [(ts, pkt), ...]

    reader = dpkt.pcap.Reader(open(input_file, 'rb'))
    for ts, pkt in reader:
        if flow_end_ts != -1 and ts - flow_end_ts > time_interval:
            # start of a new flow
            flows.append(flow)
            flow = [(ts, pkt)]
        else:
            flow.append((ts, pkt))

        flow_end_ts = ts

    if len(flow) != 0:
        flows.append(flow)

    for flow in flows:
        if len(flow) >= args.packet_num:
            if args.dataset in ['ISCX_VPN_2016_(only_VPN)','VNAT_(only_NonVPN)']:
                linktype = DLT_RAW
                #linktype = DLT_IEEE802_11
                #linktype = DLT_LINUX_SLL
            else:
                linktype = DLT_EN10MB

            output_file = f'{output_dir}/{round(flow[0][0])}_packet_num={args.packet_num}.pcap'
            with lock:  # 确保写操作是线程安全的
                writer = dpkt.pcap.Writer(open(output_file, 'wb'), linktype=linktype)
                writer.writepkts(flow)

def process_session(session_pcap_file, session_dir, interval, args):
    split_pcap_by_time(session_pcap_file, session_dir, interval, args)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", default="ISCX_Tor_2017",
                        choices=["ISCX_VPN_2016", "ISCX_Tor_2017", "CIC_IOT_2022", "USTC_TFC2016", "ISCX_Bot_2014", "CIC_IDS_2017", "VNAT", "CSTNET_TLS_1.3"])

    parser.add_argument("--packet_num", type=int, default=16)

    args = parser.parse_args()

    p = "/data/users/lph/datasets/ISCX_Dataset/ISCX_Tor_2017/TorPcaps/Pcaps"
    all_items = glob.glob(os.path.join(p, '*'))
    dirs_label = [d for d in all_items if os.path.isdir(d)]

    tgt_dir = os.path.join("/data/users/lph/projects/WTMamba/datasets/ISCX_Tor_2017_test","pcap")
    # if os.path.exists(tgt_dir):
    #     shutil.rmtree(tgt_dir)
    # os.mkdir(tgt_dir)

    delete_pcap = []
    for dir_label in dirs_label:
        dir_label = dir_label.split('/')[-1]
        session_dir = os.path.join(tgt_dir, dir_label)
        session_dir_abs = os.path.abspath(session_dir)
        # if os.path.exists(session_dir):
        #     shutil.rmtree(session_dir)
        # os.mkdir(session_dir)

        print("---------- {}/{}".format(p, dir_label))
        pp = os.path.join(p, dir_label)
        label_pcap_files = glob.glob(os.path.join(pp, '**', '*.pcap'), recursive=True)

        for file in label_pcap_files:
            if file.endswith('.pcapng'):
                shutil.move(file, file.replace('.pcapng', '.pcap'))
                file = file.replace('.pcapng', '.pcap')
            if not file.endswith('.pcap'):
                continue

            org_file_abs = os.path.abspath(file)

            os.system(f"mono /data/users/lph/tools/SplitCap.exe -r {org_file_abs} -s session -o {session_dir_abs}")

        session_pcap_files = glob.glob(os.path.join(session_dir, '*.pcap'), recursive=True)
        delete_pcap += session_pcap_files

        # 使用线程池处理拆分
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_session, session_pcap_file, session_dir, interval_seconds[args.dataset], args): session_pcap_file for session_pcap_file in session_pcap_files}
            for future in as_completed(futures):
                try:
                    future.result()  # 获取结果以捕获异常
                except Exception as e:
                    print(f"处理文件时出错: {futures[future]} - {e}")

    # # 删除临时pcap文件
    # for p in delete_pcap:
    #     try:
    #         os.remove(p)
    #         print(f"删除文件成功: {p}")
    #         time.sleep(0.5)
    #     except OSError as e:
    #         print(f"删除文件失败: {p} - {e}")
    #         continue
    #     except Exception as e:
    #         print(f"发生未知错误: {p} - {e}")

if __name__ == "__main__":
    main()