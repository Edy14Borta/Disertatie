import re
from datetime import datetime

auth_pattern = re.compile(
    r"(?P<month>\w{3})\s+(?P<day>\d+)\s+(?P<time>\d{2}:\d{2}:\d{2}).*sshd\[\d+\]:\s+(?P<status>Accepted|Failed)\s+(?P<method>\S+)\s+for\s+(?P<user>\S+)\s+from\s+(?P<ip>\S+)\s+port\s+(?P<port>\d+)"
)

last_connection_time = {}

def extract_features(log_line):
    match = auth_pattern.search(log_line)
    if not match:
        return None, None
    
    data = match.groupdict()
    current_time = datetime.strptime(data["time"], "%H:%M:%S")
    
    minutes_of_day = current_time.hour * 60 + current_time.minute
    
    ip = data["ip"]
    ip_is_external = 0 if ip.startswith("192.168.") or ip.startswith("10.") or ip in ["127.0.0.1", "::1"] else 1
    
    user = data["user"]
    high_frequency = 0
    if user in last_connection_time:
        diff = (current_time - last_connection_time[user]).total_seconds()
        if abs(diff) < 10:
            high_frequency = 1
    last_connection_time[user] = current_time
    
    auth_method = 1 if data["method"] == "password" else 0
    status = 1 if data["status"] == "Failed" else 0
    
    features = [minutes_of_day, ip_is_external, high_frequency, auth_method, status]
    
    label = 0
    if "#[1]" in log_line:
        label = 1
        
    return features, label
