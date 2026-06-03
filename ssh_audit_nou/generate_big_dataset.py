import random
import datetime

NORMAL_IPS = ["192.168.1.15", "192.168.1.16", "192.168.1.45"]
ATTACK_IPS = ["45.33.22.11", "185.220.101.30", "203.0.113.5", "198.51.100.7"]
USERS = ["root", "admin", "test", "stefan", "eduard", "user"]

TOTAL_RECORDS = 100000
NORMAL_RATIO = 0.88
FILE_NAME = "synthetic_auth.log"

def generate_timestamp(is_normal):
    base_date = datetime.datetime.now() - datetime.timedelta(days=random.randint(0, 5))
    if is_normal:
        if random.random() < 0.97:
            hour = random.randint(8, 17)
        else:
            hour = random.choice([23, 0, 1, 2, 3, 4])
    else:
        if random.random() < 0.7:
            hour = random.choice([23, 0, 1, 2, 3, 4])
        else:
            hour = random.randint(8, 17)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return base_date.replace(hour=hour, minute=minute, second=second).strftime("%b %d %H:%M:%S")

print("Generare dataset sintetic auth.log decuplat de reguli statice...")
with open(FILE_NAME, "w") as f:
    for _ in range(TOTAL_RECORDS):
        is_normal = random.random() < NORMAL_RATIO
        timestamp = generate_timestamp(is_normal)
        pid = random.randint(2000, 25000)
        
        if is_normal:
            ip = random.choice(NORMAL_IPS)
            user = random.choice(["stefan", "eduard"])
            if random.random() < 0.05:
                log_line = f"{timestamp} ubuntu sshd[{pid}]: Failed password for {user} from {ip} port {random.randint(49152, 65535)} ssh2 #[0]\n"
            else:
                log_line = f"{timestamp} ubuntu sshd[{pid}]: Accepted publickey for {user} from {ip} port {random.randint(49152, 65535)} ssh2 #[0]\n"
        else:
            if random.random() < 0.02:
                ip = random.choice(NORMAL_IPS)
            else:
                ip = random.choice(ATTACK_IPS)
            user = random.choice(USERS)
            if random.random() < 0.95:
                log_line = f"{timestamp} ubuntu sshd[{pid}]: Failed password for {user} from {ip} port {random.randint(49152, 65535)} ssh2 #[1]\n"
            else:
                log_line = f"{timestamp} ubuntu sshd[{pid}]: Accepted password for {user} from {ip} port {random.randint(49152, 65535)} ssh2 #[1]\n"
        f.write(log_line)
print("Fisier generat cu succes cu amprente logice!")
