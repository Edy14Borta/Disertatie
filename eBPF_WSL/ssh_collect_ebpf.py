from bcc import BPF
import csv
import os
import time


ebpf_code = """
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

struct event_data {
    u32 pid;
    u32 uid;
    u32 gid;
    u64 len;
    char comm[TASK_COMM_LEN];
};

BPF_PERF_OUTPUT(ssh_events);

int kprobe__sys_write(struct pt_regs *ctx, int fd, const char *buf, size_t count) {
    struct event_data data = {};
    
    data.pid = bpf_get_current_pid_tgid() >> 32;
    data.uid = bpf_get_current_uid_gid();
    data.gid = bpf_get_current_uid_gid() >> 32;
    data.len = count; // Dimensiunea bufferului scris (LEN)
    bpf_get_current_comm(&data.comm, sizeof(data.comm));

    ssh_events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}
"""


b = BPF(text=ebpf_code)
CSV_PATH = "/mnt/d/disertatie/eBPF_WSL/dataset_eBPF.csv"


os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)


file_exists = os.path.exists(CSV_PATH)
f = open(CSV_PATH, mode="a", newline="")
writer = csv.writer(f)


if not file_exists:
    writer.writerow(["pid", "uid", "gid", "len", "hour", "is_hydra", "label"]) 
    

print("Programul eBPF rulează. Datele se salvează în: {CSV_PATH}")


def callback_eveniment(cpu, data, size):
    event = b["ssh_events"].event(data)
    comanda = event.comm.decode('utf-8', errors='ignore').strip()
    
    current_hour = time.localtime().tm_hour
    
    if "hydra" in comanda or "sshd" in comanda:
        label = 1
        is_hydra = 1
    else:
        label = 0
        is_hydra = 0
        
    writer.writerow([event.pid, event.uid, event.gid, event.len, current_hour, is_hydra, label])
    #f.flush()

b["ssh_events"].open_perf_buffer(callback_eveniment)

try:
    while True:
        b.perf_buffer_poll()
except KeyboardInterrupt:
    print("Colectare oprită cu succes. Fișierul CSV a fost salvat.")
    f.close()
