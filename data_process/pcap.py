'''
Handle the raw dataset: raw_pcap -> pcap
* split flow by session & interval time
'''
import argparse
import os
import dpkt
from dpkt.pcap import DLT_RAW, DLT_EN10MB
import shutil
import glob
import time
# clean_protocols = '"not arp and not (dns or mdns) and not stun and not dhcpv6 and not icmpv6 and not icmp and not dhcp and not llmnr and not nbns and not ntp and not igmp"'

interval_seconds = {
    'ISCXVPN2016': 0.128,
    'ISCX_Tor_2017': 0.256,
    'ISCXTor2016':0.128
}

def split_pcap_by_time(input_file, output_dir, time_interval, args):
    # time_interval = time_interval * 1e9 // 16384 * 16384 / 1e9  # taking into account the precision of tofino operations
    flow_end_ts = -1
    flows = []  # [flow, ...]#存储每条流
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

    #save_flow=[]
    for flow in flows:
        if len(flow)>=args.packet_num:
            #print("合格的流")
            #save_flow.append(flow)
            if args.dataset in ['ISCX_Tor_2017']:
                linktype = DLT_RAW
            else:
                linktype = DLT_EN10MB
            writer = dpkt.pcap.Writer(
                open(output_dir + '/{}_packet_num={}.pcap'.format(round(flow[0][0]),args.packet_num), 'wb'), linktype=linktype
            )
            writer.writepkts(flow)
        else:
            pass


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Dataset
    parser.add_argument("--dataset", default="ISCX_Tor_2017",
                        choices=["ISCXVPN2016","ISCXTor2016","ISCX_Tor_2017"])

    parser.add_argument("--packet_num", type=int ,default="10",)#一条流中，数据包的最少个数

    args = parser.parse_args()
    p = "/data/users/lph/datasets/ISCX_Dataset/ISCX_Tor_2017/TorPcaps/Pcaps"
    all_items = glob.glob(os.path.join(p, '*'))
    dirs_label = [d for d in all_items if os.path.isdir(d)]
    delete_pcap = []
    #for p, dirs_label, _ in os.walk('../datasets/{}'.format(args.dataset)):
    # tgt_dir = os.path.join(p.replace('raw', ''), "pcap").replace('\\', '/')
    tgt_dir = os.path.join("/data/users/lph/projects/WTMamba/datasets/ISCX_Tor_2017", "pcap")
    if os.path.exists(tgt_dir):
        shutil.rmtree(tgt_dir)
    os.mkdir(tgt_dir)
    for dir_label in dirs_label:
        dir_label = dir_label.split('/')[-1]
        session_dir = os.path.join(tgt_dir, dir_label)
        # session_dir = clean_file.replace('.pcap', '')
        session_dir_abs = os.path.abspath(session_dir)
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
        os.mkdir(session_dir)

        print("---------- {}/{}".format(p, dir_label))
        pp = os.path.join(p, dir_label)
        label_pcap_files = glob.glob(os.path.join(pp, '*.pcap'))

        for file in label_pcap_files:
            # rename pcapng as pcap
            if file.find('.pcapng') != -1:
                shutil.move(os.path.join(pp, file), os.path.join(pp, file.replace('.pcapng', '.pcap')))
                file = file.replace('.pcapng', '.pcap')
            if file.find('.pcap') == -1:
                continue


            org_file_abs = os.path.abspath(file)
            os.system(f" mono /data/users/lph/tools/SplitCap.exe -r {org_file_abs} -s session -o {session_dir_abs}")

        #for _, _, session_pcap_files in os.walk(session_dir):
        session_pcap_files = glob.glob(os.path.join(session_dir, '*.pcap'))
        delete_pcap += session_pcap_files
        for session_pcap_file in session_pcap_files:
            #session_pcap_file = os.path.join(session_dir, session_pcap_file)
            #segment_dir = session_pcap_file.replace(file + '.', '').replace('.pcap', '')
            split_pcap_by_time(session_pcap_file, session_dir, interval_seconds[args.dataset], args)
            #os.remove(session_pcap_file)
    #删除seesion pcap
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