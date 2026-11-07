"""Gazetteers for rule-based NER: {entity_type: {canonical_name: regex_alias_alternation}}.

* Matching is case-insensitive by default; `(?-i:...)` makes a pattern case-sensitive (used for ambiguous common words
  such as Windows, Java, Exchange, Oracle, React).
* Canonical names are the node labels of the knowledge graph (entity normalisation + deduplication of aliases).
* To extend the NER, add entries here: no other code change is needed (see docs/PROJECT_DEFENSE_GUIDE.md).
* ATTACK = attack technique OR malware class (ransomware, rootkit...); MALWARE = named malware family;
  VULN_NAME = named vulnerability/exploit (Heartbleed, EternalBlue...) that the reference KB maps to a CVE.
"""

CS = "(?-i:{})"   # case-sensitive wrapper


def cs(p: str) -> str:
    return CS.format(p)


LEXICON: dict[str, dict[str, str]] = {
    "MALWARE": {
        "WannaCry": "wannacry|wanacry", "Emotet": "emotet", "TrickBot": "trickbot", "Mirai": "mirai", "Stuxnet": "stuxnet",
        "LockBit": "lockbit", "Ryuk": "ryuk", "NotPetya": "notpetya|petya", "Cobalt Strike": "cobalt strike",
        "Mimikatz": "mimikatz", "QakBot": "qakbot|qbot", "Zeus": r"zeus(?:\s+trojan)?", "Conti": cs(r"Conti"),
        "REvil": "revil|sodinokibi", "Agent Tesla": "agent tesla", "DarkSide": "darkside", "BlackEnergy": "blackenergy",
        "CryptoLocker": "cryptolocker", "SUNBURST": "sunburst", "Duqu": "duqu", "ZeroAccess": "zeroaccess",
        "njRAT": "njrat", "Carbanak": "carbanak", "Maze": cs(r"Maze"),
    },
    "THREAT_ACTOR": {
        "APT28": r"apt\s?28|fancy bear|sofacy", "APT29": r"apt\s?29|cozy bear", "Lazarus Group": r"lazarus(?:\s+group)?",
        "Turla": cs("Turla"), "Sandworm": "sandworm", "APT41": r"apt\s?41", "FIN7": "fin7", "Equation Group": "equation group",
        "OceanLotus": r"oceanlotus|apt\s?32",
    },
    "SOFTWARE": {
        "Windows": cs(r"Windows(?:\s+(?:10|11|7|8|XP|Server(?:\s+\d{4})?))?"), "Linux": r"linux", "Android": r"android",
        "iOS": cs(r"iOS"), "macOS": r"macos|mac os x|os x", "Chrome": cs(r"Chrome|Chromium"), "Firefox": r"firefox",
        "Internet Explorer": r"internet explorer|msie", "Microsoft Edge": r"microsoft edge", "Safari": cs(r"Safari"),
        "Apache Struts": r"apache struts|struts2?", "Apache Tomcat": r"tomcat", "Apache Log4j": r"log4j|log4shell",
        "Apache HTTP Server": r"apache(?:\s+http(?:d|\s+server)|\s+web\s+server)|(?-i:Apache)(?!\s+(?:Struts|Tomcat|CXF|Xerces|Kafka|Spark|Cassandra|Log4j|Software))",
        "Nginx": "nginx", "OpenSSL": "openssl", "OpenSSH": "openssh", "Samba": cs(r"Samba"), "MySQL": "mysql",
        "PostgreSQL": r"postgres(?:ql)?", "MongoDB": "mongodb", "Redis": cs(r"Redis"), "Elasticsearch": "elasticsearch",
        "Jenkins": cs(r"Jenkins"), "Docker": "docker", "Kubernetes": r"kubernetes|k8s", "VMware": "vmware",
        "Hyper-V": r"hyper-v", "VirtualBox": "virtualbox", "WordPress": "wordpress",
        "Microsoft Exchange": r"(?:microsoft|ms)\s+exchange(?:\s+server)?|exchange\s+server", "SharePoint": "sharepoint",
        "Active Directory": r"active directory", "Oracle Database": cs(r"Oracle(?:\s+(?:Database|DB))?"), "Java": cs(r"Java"),
        "PHP": cs(r"PHP"), "Python": cs(r"Python"), "JavaScript": cs(r"JavaScript|JS"), "QEMU": cs(r"QEMU"), "runC": cs(r"runC|runc"), "Rust": cs(r"Rust"), "Go": cs(r"Golang"), "PowerShell": r"powershell", "Node.js": r"node\.?js",
        "React": cs(r"React(?:\.js|JS)?"), "Angular": cs(r"Angular(?:JS)?"), "jQuery": "jquery", "curl": r"libcurl|(?-i:\bcurl\b)",
        "glibc": "glibc", "Bash": cs(r"Bash"), ".NET Framework": r"\.net framework", "GitHub": cs(r"GitHub"),
        "AWS": cs(r"AWS|Amazon Web Services"), "Azure": cs(r"Azure"), "Microsoft Office": r"microsoft office|ms office",
        "Adobe Flash": r"adobe flash|flash player", "WebView": cs(r"WebView"), "Kali Linux": r"kali linux",
    },
    "TOOL": {
        "Wireshark": "wireshark", "Metasploit": "metasploit", "Burp Suite": r"burp(?:\s+suite)?", "Nmap": "nmap",
        "Splunk": "splunk", "Volatility": cs(r"Volatility"), "Ghidra": "ghidra", "IDA Pro": cs(r"IDA(?:\s+Pro)?"),
        "WinDbg": "windbg", "Sysinternals": "sysinternals", "PsExec": "psexec", "YARA": cs(r"YARA"), "sqlmap": "sqlmap",
        "OWASP ZAP": cs(r"ZAP|OWASP ZAP"), "Nessus": "nessus", "Hashcat": "hashcat", "John the Ripper": "john the ripper",
        "BloodHound": "bloodhound", "Snort": cs(r"Snort"), "Suricata": "suricata", "Zeek": cs(r"Zeek"), "Wazuh": "wazuh",
        "Responder": cs(r"Responder"), "Sysmon": "sysmon", "ROPgadget": "ropgadget", "BITSAdmin": "bitsadmin", "VMProtect": "vmprotect", "UPX": cs(r"UPX"), "x64dbg": "x64dbg", "OllyDbg": "ollydbg", "Frida": cs(r"Frida"), "radare2": "radare2", "Cuckoo Sandbox": r"cuckoo", "Nagios": "nagios", "Zabbix": "zabbix", "Autopsy": cs(r"Autopsy"), "Process Monitor": "process monitor|procmon",
    },
    "TECHNOLOGY": {
        "TLS/SSL": r"\btls\b|\bssl\b|tls\s?1\.[0-3]", "HTTPS": r"https", "DNS": cs(r"DNS"), "DNSSEC": cs(r"DNSSEC"),
        "Kerberos": "kerberos", "NTLM": cs(r"NTLM(?:v2)?"), "SMB": cs(r"SMB(?:v[123])?"), "RDP": cs(r"RDP"), "SSH": cs(r"SSH"),
        "OAuth": r"oauth\s?2?(?:\.0)?", "OpenID Connect": r"openid connect|oidc", "SAML": cs(r"SAML"), "JWT": cs(r"JWTs?"),
        "LDAP": cs(r"LDAP"), "Telnet": cs(r"Telnet"), "VPN": cs(r"VPN"), "WPA3": cs(r"WPA3"), "WPA2": cs(r"WPA2"), "Bluetooth": "bluetooth",
        "GraphQL": "graphql", "PKI": cs(r"PKI"), "RSA": cs(r"RSA(?:-\d{3,4})?"), "AES": cs(r"AES(?:-\d{3})?"),
        "ECC/ECDSA": cs(r"ECC|ECDSA|ECDH"), "Diffie-Hellman": r"diffie-hellman", "SHA-256": r"sha-?256", "SHA-1": r"sha-?1\b",
        "MD5": cs(r"MD5"), "Intel": cs(r"Intel"), "AMD": cs(r"AMD"), "ARM": cs(r"ARM"), "Intel SGX": cs(r"SGX"),
        "UEFI": cs(r"UEFI"), "XPC": cs(r"XPC"), "ETW": cs(r"ETW"), "AMSI": cs(r"AMSI"), "WMI": cs(r"WMI"), "LoRaWAN": cs(r"LoRaWAN"), "Zigbee": r"zigbee", "CAN bus": r"can bus", "5G": cs(r"5G"), "BGP": cs(r"BGP"), "BIOS": cs(r"BIOS"), "TPM": cs(r"TPM"), "Bitcoin": cs(r"Bitcoin|Blockchain"),
        "REST API": r"rest(?:ful)?\s+api",
    },
    "ATTACK": {
        "reflected XSS": r"reflected xss|reflected cross-site", "stored XSS": r"stored xss|persistent xss|stored cross-site",
        "DOM XSS": r"dom[- ]based xss|dom xss", "XSS": r"\bxss\b|cross[- ]site scripting",
        "SQL injection": r"sql injection|sqli", "command injection": r"command injection|os command",
        "CSRF": r"\bcsrf\b|cross[- ]site request forgery", "SSRF": r"\bssrf\b|server[- ]side request forgery",
        "XXE": r"\bxxe\b|xml external entit", "phishing": r"phishing|spear[- ]phishing|smishing|vishing",
        "DDoS": r"\bddos\b|denial[- ]of[- ]service", "MITM": r"\bmitm\b|man[- ]in[- ]the[- ]middle|adversary-in-the-middle",
        "ransomware": "ransomware", "brute force": r"brute[- ]force|credential stuffing|password spraying",
        "RCE": r"\brce\b|remote code execution", "privilege escalation": r"privilege escalation|escalat\w* (?:their |its |the |his )?privileges?|elevat\w* privileges|\bLPE\b",
        "buffer overflow": r"buffer overflow|stack overflow attack|heap overflow", "directory traversal": r"directory traversal|path traversal",
        "session hijacking": "session hijack", "DOM clobbering": "dom clobbering", "clickjacking": "clickjacking",
        "DNS spoofing": r"dns (?:spoof|poison)", "zero-day": r"zero[- ]day", "social engineering": "social engineering",
        "botnet": "botnet", "keylogger": "keylogger", "rootkit": "rootkit", "trojan": "trojan", "backdoor": "backdoor",
        "lateral movement": "lateral movement", "ROP": r"\brop\b|return[- ]oriented programming", "JOP": cs(r"JOP"),
        "heap spray": "heap spray",  "padding oracle": "padding oracle", "prototype pollution": "prototype pollution", "stack pivoting": "stack pivot", "signature wrapping": "signature wrapping", "memory corruption": "memory corruption", "CRIME": cs(r"CRIME"), "BREACH": cs(r"BREACH"), "BEAST": cs(r"BEAST"), "return-to-libc": r"ret(?:urn)?[- ]?2?[- ]?to[- ]?libc|ret2libc", "cache attack": r"cache[- ](?:timing |side-channel )?attack", "Rowhammer": "rowhammer", "cold boot attack": "cold boot", "fault injection": "fault injection", "use-after-free": r"use[- ]after[- ]free|\buaf\b", "type confusion": "type confusion",
        "format string": "format string", "integer overflow": "integer overflow", "race condition": "race condition|toctou",
        "DLL hijacking": r"dll (?:hijack|side-load|search order)", "process injection": r"process injection|process hollowing|dll injection|hollowing|inject\w* (?:\w+ )?into (?:an? |the )?(?:existing |legitimate |running |remote |target )?process|apc injection",
        "pass-the-hash": r"pass[- ]the[- ]hash", "pass-the-ticket": r"pass[- ]the[- ]ticket", "golden ticket": "golden ticket",
        "Kerberoasting": "kerberoasting", "credential dumping": "credential dumping", "side-channel": r"side[- ]channel|timing attack",
        "sandbox escape": r"sandbox escape|vm escape|virtual machine escape|virtuali[sz]ation escape|break(?:ing)? out of the sandbox|escap\w* (?:from )?the sandbox|hypervisor escape", "replay attack": "replay attack",
        "supply chain attack": r"supply[- ]chain (?:attack|compromise)", "watering hole": "watering hole",
        "APT": cs(r"APTs?"), "IDOR": cs(r"IDOR"), "LFI/RFI": cs(r"LFI|RFI"),
        "open redirect": "open redirect", "cryptojacking": "cryptojacking", "spyware": "spyware", "worm": r"\bworms?\b",
        "port scanning": r"port scan", "ARP spoofing": r"arp (?:spoof|poison)", "DLL proxying": "dll proxying",
        "token manipulation": r"token (?:manipulation|impersonation|theft)",
    },
    "VULN_NAME": {
        "Heartbleed": "heartbleed", "Shellshock": "shellshock", "EternalBlue": "eternalblue", "BlueKeep": "bluekeep",
        "Zerologon": "zerologon", "PrintNightmare": "printnightmare", "Log4Shell": "log4shell", "Dirty COW": r"dirty\s?cow",
        "Spectre": cs(r"Spectre"), "Meltdown": cs(r"Meltdown"), "KRACK": cs(r"KRACK"), "POODLE": cs(r"POODLE"),
        "Stagefright": "stagefright", "Foreshadow": "foreshadow|l1tf",
    },
    "DEFENSE": {
        "ASLR": cs(r"ASLR|KASLR"), "DEP/NX": cs(r"DEP|NX bit|W\^X"), "stack canary": r"stack (?:canary|cookie|protector)|canaries",
        "CFI/CFG": cs(r"CFI|CFG|Control[- ]Flow (?:Integrity|Guard)"), "SMEP/SMAP": cs(r"SMEP|SMAP"), "SEHOP": cs(r"SEHOP"),
        "CSP": r"content security policy|\bcsp\b", "WAF": cs(r"WAFs?"), "MFA": r"\bmfa\b|\b2fa\b|multi-?factor", "EDR": cs(r"EDR"),
        "SIEM": cs(r"SIEMs?"), "IDS/IPS": cs(r"IDS|IPS"), "SELinux": cs(r"SELinux"), "AppLocker": cs(r"AppLocker"),
        "Windows Defender": r"(?:windows|microsoft) defender", "SameSite cookies": "samesite", "HttpOnly": cs(r"HttpOnly"),
        "CORS": cs(r"CORS"), "Secure Boot": "secure boot", "sandboxing": r"sandboxing|sandboxed", "input validation": "input validation",
        "output encoding": r"output encoding|output escaping", "parameterized queries": r"parameteri[sz]ed quer|prepared statement",
        "least privilege": "least privilege", "network segmentation": "network segmentation", "rate limiting": "rate limit",
        "zero trust": "zero trust", "code signing": "code signing", "patch management": r"patch management|patching",
        "RBAC": cs(r"RBAC"), "YARA rules": "yara rule",
    },
    "TARGET": {
        "web application": r"web app(?:lication)?s?", "API": cs(r"APIs?"), "database": r"\bdatabases?\b|\bdbms?\b",
        "browser": r"\bbrowsers?\b", "email": r"\be-?mails?\b", "credentials": r"credentials?|passwords?",
        "cloud": r"\bcloud\b", "IoT device": r"\biot\b|embedded device", "mobile device": r"mobile (?:app|device)s?|smartphones?",
        "kernel": r"\bkernel\b", "hypervisor": r"hypervisor|virtual machines?", "container": r"\bcontainers?\b",
        "firewall": r"firewalls?", "network": r"\bnetworks?\b", "endpoint": r"\bendpoints?\b|workstation", "server": r"\bservers?\b",
        "user": r"\busers?\b|employees?|victims?", "domain controller": "domain controller",
    },
}

MITRE_TACTICS = ["Initial Access", "Execution", "Persistence", "Privilege Escalation", "Defense Evasion", "Credential Access",
                 "Discovery", "Lateral Movement", "Collection", "Command and Control", "Exfiltration", "Impact",
                 "Reconnaissance", "Resource Development"]

# ATT&CK technique id -> name (subset I am certain about; ids in the corpus that are absent here are deprecated/ICS ids)
MITRE_NAMES = {
    "T1003": "OS Credential Dumping", "T1003.001": "LSASS Memory", "T1005": "Data from Local System", "T1021": "Remote Services",
    "T1027": "Obfuscated Files or Information", "T1041": "Exfiltration Over C2 Channel", "T1047": "Windows Management Instrumentation",
    "T1053": "Scheduled Task/Job", "T1055": "Process Injection", "T1055.012": "Process Hollowing", "T1059": "Command and Scripting Interpreter",
    "T1059.001": "PowerShell", "T1059.004": "Unix Shell", "T1068": "Exploitation for Privilege Escalation", "T1070": "Indicator Removal",
    "T1071": "Application Layer Protocol", "T1078": "Valid Accounts", "T1082": "System Information Discovery", "T1090": "Proxy",
    "T1095": "Non-Application Layer Protocol", "T1105": "Ingress Tool Transfer", "T1110": "Brute Force", "T1112": "Modify Registry",
    "T1133": "External Remote Services", "T1134": "Access Token Manipulation", "T1140": "Deobfuscate/Decode Files or Information",
    "T1189": "Drive-by Compromise", "T1190": "Exploit Public-Facing Application", "T1195": "Supply Chain Compromise",
    "T1203": "Exploitation for Client Execution", "T1204": "User Execution", "T1210": "Exploitation of Remote Services",
    "T1218": "System Binary Proxy Execution", "T1486": "Data Encrypted for Impact", "T1497": "Virtualization/Sandbox Evasion",
    "T1498": "Network Denial of Service", "T1499": "Endpoint Denial of Service", "T1543": "Create or Modify System Process",
    "T1547": "Boot or Logon Autostart Execution", "T1548": "Abuse Elevation Control Mechanism", "T1550": "Use Alternate Authentication Material",
    "T1557": "Adversary-in-the-Middle", "T1558": "Steal or Forge Kerberos Tickets", "T1562": "Impair Defenses", "T1565": "Data Manipulation",
    "T1566": "Phishing", "T1569": "System Services", "T1573": "Encrypted Channel", "T1574": "Hijack Execution Flow",
}

# generic nouns: kept in the graph but flagged as low-information (they co-occur with almost everything)
LOW_INFORMATION = {"TARGET"}
