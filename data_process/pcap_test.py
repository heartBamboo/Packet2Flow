import dpkt

def get_linktype(pcap_file):
    with open(pcap_file, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        return pcap.datalink()

# 示例用法
#input_file = '/data/users/lph/datasets/ISCX_Dataset/ISCX_Tor_2017/TorPcaps/Pcaps/tor/CHAT_hangoutschatgateway.pcap'
input_file = '/data/users/lph/datasets/VNAT/Pcap/VPN_C2/vpn_ssh_capture2.pcap'
#input_file = '/data/users/lph/datasets/CSTNET_TLS_1.3/finetune/pcap/cstnet-tls 1.3/overleaf.com/1.pcap'
linktype = get_linktype(input_file)
print(f"Link type: {linktype}")