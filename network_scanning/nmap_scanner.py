import nmap
from scapy.all import ARP, Ether, srp, TCP, IP, sr1, ICMP
from config import SCAN_TYPES
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class NmapScanner:
    def __init__(self):
        self.scanner = nmap.PortScanner()

    def scan_network(self, target_ip, scan_type, custom_ports=None):
        """
        Perform a network scan using Nmap.
        Supports all scan types defined in SCAN_TYPES.
        """
        try:
            # Get the Nmap arguments for the selected scan type
            arguments = SCAN_TYPES.get(scan_type, "-T4 -A -v")
            
            # Replace {ports} placeholder with custom ports if provided
            if custom_ports and "{ports}" in arguments:
                arguments = arguments.format(ports=custom_ports)
            
            logging.info(f"Starting {scan_type} scan on {target_ip} with arguments: {arguments}")
            
            # Perform the Nmap scan
            scan_result = self.scanner.scan(hosts=target_ip, arguments=arguments)
            logging.info(f"Scan completed for {target_ip}")
            
            return scan_result
        except Exception as e:
            logging.error(f"Error during Nmap scan: {e}")
            raise

    def arp_scan(self, target_ip):
        """
        Perform an ARP scan to discover live hosts on the network using Scapy.
        """
        try:
            logging.info(f"Starting ARP scan on {target_ip}")
            # Create ARP request packet
            arp_request = ARP(pdst=target_ip)
            ether = Ether(dst="ff:ff:ff:ff:ff:ff")
            packet = ether / arp_request

            # Send packet and receive response
            result = srp(packet, timeout=2, verbose=0)[0]

            # Parse the response
            live_hosts = []
            for sent, received in result:
                live_hosts.append({"IP": received.psrc, "MAC": received.hwsrc})

            logging.info(f"ARP scan completed. Found {len(live_hosts)} live hosts.")
            return live_hosts
        except Exception as e:
            logging.error(f"Error during ARP scan: {e}")
            raise

    def tcp_syn_scan(self, target_ip, ports):
        """
        Perform a TCP SYN scan using Scapy to check for open ports.
        """
        try:
            logging.info(f"Starting TCP SYN scan on {target_ip} for ports {ports}")
            open_ports = []
            for port in ports:
                try:
                    # Create SYN packet
                    syn_packet = IP(dst=target_ip) / TCP(dport=port, flags="S")
                    response = sr1(syn_packet, timeout=1, verbose=0, retry=2)

                    if response and response.haslayer(TCP):
                        if response.getlayer(TCP).flags == 0x12:  # SYN-ACK
                            open_ports.append(port)
                            # Send RST to close the connection
                            rst_packet = IP(dst=target_ip) / TCP(dport=port, flags="R")
                            sr1(rst_packet, timeout=1, verbose=0)
                except Exception as e:
                    logging.warning(f"Error scanning port {port}: {e}")
                    continue

            logging.info(f"TCP SYN scan completed. Open ports: {open_ports}")
            return open_ports
        except Exception as e:
            logging.error(f"Error during TCP SYN scan: {e}")
            raise

    def icmp_ping_sweep(self, target_ip):
        """
        Perform an ICMP ping sweep to check for live hosts using Scapy.
        """
        try:
            logging.info(f"Starting ICMP ping sweep on {target_ip}")
            # Create ICMP echo request packet
            icmp_packet = IP(dst=target_ip) / ICMP()
            response = sr1(icmp_packet, timeout=1, verbose=0)

            if response:
                logging.info(f"Host {target_ip} is up.")
                return True
            else:
                logging.info(f"Host {target_ip} is down.")
                return False
        except Exception as e:
            logging.error(f"Error during ICMP ping sweep: {e}")
            raise

    def os_fingerprinting(self, target_ip):
        """
        Perform OS fingerprinting using Nmap.
        """
        try:
            logging.info(f"Starting OS fingerprinting on {target_ip}")
            scan_result = self.scanner.scan(hosts=target_ip, arguments="-O")
            os_info = scan_result["scan"][target_ip].get("osmatch", [])
            if os_info:
                logging.info(f"OS fingerprinting completed. Detected OS: {os_info[0]['name']}")
                return os_info
            else:
                logging.info("No OS information detected.")
                return None
        except Exception as e:
            logging.error(f"Error during OS fingerprinting: {e}")
            raise

    def vulnerability_scan(self, target_ip):
        """
        Perform a vulnerability scan using Nmap NSE scripts.
        """
        try:
            logging.info(f"Starting vulnerability scan on {target_ip}")
            scan_result = self.scanner.scan(hosts=target_ip, arguments="--script=vuln")
            vulnerabilities = scan_result["scan"][target_ip].get("tcp", {})
            logging.info(f"Vulnerability scan completed. Found {len(vulnerabilities)} potential vulnerabilities.")
            
            # Transform vulnerabilities into a list of dictionaries
            vuln_list = []
            for port, port_info in vulnerabilities.items():
                if "script" in port_info:
                    for script_name, script_output in port_info["script"].items():
                        vuln_list.append({
                            "Port": port,
                            "Vulnerability": script_name,
                            "Description": script_output
                        })
            
            return vuln_list
        except Exception as e:
            logging.error(f"Error during vulnerability scan: {e}")
            raise

    def service_version_detection(self, target_ip):
        """
        Perform service version detection using Nmap.
        """
        try:
            logging.info(f"Starting service version detection on {target_ip}")
            scan_result = self.scanner.scan(hosts=target_ip, arguments="-sV")
            services = scan_result["scan"][target_ip].get("tcp", {})
            logging.info(f"Service version detection completed. Found {len(services)} services.")
            
            # Transform services into a list of dictionaries
            service_list = []
            for port, port_info in services.items():
                service_list.append({
                    "Port": port,
                    "Service": port_info.get("name", "N/A"),
                    "Product": port_info.get("product", "N/A"),
                    "Version": port_info.get("version", "N/A"),
                    "Extra Info": port_info.get("extrainfo", "N/A"),
                    "CPE": port_info.get("cpe", "N/A")
                })
            
            return service_list
        except Exception as e:
            logging.error(f"Error during service version detection: {e}")
            raise