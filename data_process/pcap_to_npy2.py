'''
Preprocess the .pcap file of flows: .pcap -> .npy
'''
import argparse
import json
import os
import shutil
import numpy as np
import sys

sys.path.append('..')
from scapy.all import rdpcap
from scapy.all import *


def process_packet(pkt,args):
    # 检查数据包是否有 IP 层
    if IP in pkt:
        ip_layer = pkt[IP]

        # 创建一个新的 IP 头部，只保留前 20 个字节
        new_ip_header = IP(
            version=ip_layer.version,
            ihl=5,  # 固定为 5 (20 字节)
            tos=ip_layer.tos,
            len=args.payload_len+48,  # 总长度为 256 字节
            id=ip_layer.id,
            flags=ip_layer.flags,
            frag=ip_layer.frag,
            ttl=ip_layer.ttl,
            proto=ip_layer.proto,
            chksum=None,  # 让 Scapy 自动计算校验和
            src='0.0.0.0',  # 去除源地址
            dst='0.0.0.0'  # 去除目标地址
        )

        # 检查是 TCP 还是 UDP 协议
        if TCP in pkt:
            tcp_layer = pkt[TCP]

            # 创建一个新的 TCP 头部，只保留前 20 个字节
            new_tcp_header = TCP(
                sport=0,  # 去除源端口
                dport=0,  # 去除目标端口
                seq=tcp_layer.seq,
                ack=tcp_layer.ack,
                dataofs=5,  # 固定为 5 (20 字节)
                reserved=0,
                flags=tcp_layer.flags,
                window=tcp_layer.window,
                chksum=None,  # 让 Scapy 自动计算校验和
                urgptr=0
            )

            # 创建 UDP 头部，填充 8 个字节的 0
            new_udp_header = UDP(
                sport=0,  # 去除源端口
                dport=0,  # 去除目标端口
                len=8,  # UDP 头部加 payload 长度
                chksum=None  # 让 Scapy 自动计算校验和
            )

            # 创建新的数据包
            payload = bytes(pkt[TCP].payload)
            if len(payload) < args.payload_len:
                #print('Payload too short: ' + str(len(payload)))
                payload =payload + b'\x00' * (args.payload_len - len(payload))
            new_payload = payload[:args.payload_len]
            #new_payload = pkt[TCP].payload[:208]  #  保留 header+payload 字节的 payload
            #new_pkt = new_ip_header / new_tcp_header / new_payload / Raw(load=b'\x00' * 8)
            new_pkt = new_ip_header / new_tcp_header / Raw(load=b'\x00' * 8) / new_payload

        elif UDP in pkt:
            udp_layer = pkt[UDP]

            # 创建一个新的 UDP 头部，只保留前 8 个字节
            new_udp_header = UDP(
                sport=0,  # 去除源端口
                dport=0,  # 去除目标端口
                len=216,  # UDP 头部加 payload 长度
                chksum=None  # 让 Scapy 自动计算校验和
            )

            # 创建 TCP 头部，填充 20 个字节的 0
            new_tcp_header = TCP(
                sport=0,  # 去除源端口
                dport=0,  # 去除目标端口
                seq=0,
                ack=0,
                dataofs=5,  # 固定为 5 (20 字节)
                reserved=0,
                flags=0,
                window=0,
                chksum=None,  # 让 Scapy 自动计算校验和
                urgptr=0
            )

            # 创建新的数据包
            payload = bytes(pkt[UDP].payload)
            if len(payload) < args.payload_len:
                #print('Payload too short: ' + str(len(payload)))
                payload =payload + b'\x00' * (args.payload_len - len(payload))
            new_payload = payload[:args.payload_len] # 保留 header+payload 字节的 payload
            #new_pkt = new_ip_header / new_tcp_header / Raw(load=b'\x00' * 20) / new_udp_header / new_payload
            new_pkt = new_ip_header / Raw(load=b'\x00' * 20) / new_udp_header / new_payload
        else:
            new_pkt = None
        if(new_pkt):
            packet_bytes = bytes(new_pkt)
            decimal_array = [b for b in packet_bytes]
            decimal_array_np = np.array(decimal_array, dtype=np.uint8)
            #return decimal_array_np
            return decimal_array_np.reshape((1, -1))  # 确保返回形状为 (1, N)
        else:
            return None
    else:
        return None


def preprocess_flow(flow_file, args):
    pkts = rdpcap(flow_file)
    if len(pkts) < args.window_size:
        return None
    flow = []
    for pkt in pkts[:args.window_size]:
        new_packet = process_packet(pkt, args)
        if new_packet is not None:  # 确保 new_packet 不是 None
            flow.append(new_packet)

    if flow:
        return flow
    else:
        print(f"No valid packets found for flow file: {flow_file}")
        return None


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default="ISCX_VPN_2016_APP",
                        choices=['CIC_IOT_2022', 'ISCX_VPN_2016', 'ISCX_VPN_2016_APP', 'ISCX_Bot_2014', 'USTC_TFC2016',
                                 'CIC_IDS_2017', 'ISCX_Tor_2017', 'VNAT', 'CSTNET_TLS_1.3'])
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--payload_len", type=int, default=976, choices=[208, 528, 976])
    parser.add_argument("--packet_len", type=int, default=1024, choices=[256, 576, 1024])
    args = parser.parse_args()

    len_distribution = {}
    with open('../datasets/{}/raw/labels.json'.format(args.dataset)) as fp:
        labels = json.load(fp)

    tgt_dir = '/data/users/lph/projects/WTMamba/datasets/{}/npy'.format(args.dataset)
    if os.path.exists(tgt_dir):
        shutil.rmtree(tgt_dir)
    os.mkdir(tgt_dir)

    for label in labels.keys():
        label_dir = '/data/users/lph/projects/WTMamba/datasets/{}/npy/{}'.format(args.dataset, label)
        if os.path.exists(label_dir):
            shutil.rmtree(label_dir)
        os.mkdir(label_dir)

        one_label_flows = []
        len_distribution[label] = {}
        src_pcap_dir = '../datasets/{}/pcap/{}'.format(args.dataset, label)

        for _, _, pcap_files in os.walk(src_pcap_dir):
            for pcap_file in pcap_files:
                flow = preprocess_flow(os.path.join(src_pcap_dir, pcap_file), args)
                if flow is None or not flow:  # 检查 flow 是否为 None 或空
                    print(f"No valid flow from file: {pcap_file}, skipping...")
                    continue  # 跳过当前文件的处理
                # 检查 flow 的形状是否为 (8, 1024)
                flow_shape = np.vstack(flow).shape
                if flow_shape == (args.window_size, args.packet_len):
                    one_label_flows.append(np.vstack(flow))
                else:
                    print(f"Skipped flow with shape {flow_shape}, expected ({args.window_size}, {args.packet_len})")

        # # 确保所有流的形状一致
        # if one_label_flows:
        #     # 这里将 one_label_flows 中的每个元素都转换为 NumPy 数组
        #     one_label_flows = [np.vstack(flow) for flow in one_label_flows]
        #     shapes = [flow.shape for flow in one_label_flows]
        #     print("Shapes of valid flows:", shapes)
        #     # 这里不再使用 assert，而是直接处理有效流
        #     flow_numpy = np.vstack(one_label_flows)
        #     flow_num = flow_numpy.shape[0]
        #     np.save(tgt_dir + '/' + label + '/' + label + '_Pay=' + str(args.packet_len) + '_Win=' + str(
        #         args.window_size) + '_Num=' + str(flow_num) + '.npy', flow_numpy)
        # else:
        #     print(f"No valid flows found for label: {label}")
        if one_label_flows:
            # 将 one_label_flows 中的每个元素都转换为 NumPy 数组
            # 确保 one_label_flows 中的每个元素都是 (窗口大小, 包长)
            one_label_flows = np.array(one_label_flows)  # 转换为 NumPy 数组
            flow_numpy = one_label_flows  # 这里不再使用 vstack
            flow_num = flow_numpy.shape[0]  # 流数
            np.save(tgt_dir + '/' + label + '/' + label + '_Pay=' + str(args.packet_len) + '_Win=' + str(
                args.window_size) + '_Num=' + str(flow_num) + '.npy', flow_numpy)
        else:
            print(f"No valid flows found for label: {label}")

if __name__ == "__main__":
    main()