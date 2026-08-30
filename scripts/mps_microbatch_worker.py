#!/usr/bin/env python3
import argparse, csv, json, os, time
from pathlib import Path
import torch
p=argparse.ArgumentParser()
p.add_argument("--label",required=True); p.add_argument("--csv",required=True); p.add_argument("--summary",required=True)
p.add_argument("--start-epoch",type=float,required=True); p.add_argument("--run-seconds",type=float,required=True)
p.add_argument("--matrix-size",type=int,default=512); p.add_argument("--batch-size",type=int,default=50)
a=p.parse_args()
torch.set_float32_matmul_precision("high")
x=torch.randn((a.matrix_size,a.matrix_size),device="cuda"); y=torch.randn((a.matrix_size,a.matrix_size),device="cuda")
for _ in range(20): torch.matmul(x,y)
torch.cuda.synchronize()
while time.time()<a.start_epoch: time.sleep(.001)
path=Path(a.csv); path.parent.mkdir(parents=True,exist_ok=True)
deadline=time.monotonic()+a.run_seconds
start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
rows=[]; batch=0
with path.open("w",newline="",encoding="utf-8") as f:
 w=csv.DictWriter(f,fieldnames=["label","batch","start_unix_s","operations","wall_ms","cuda_batch_ms"]); w.writeheader()
 while time.monotonic()<deadline:
  stamp,wall=time.time(),time.perf_counter(); start.record()
  for _ in range(a.batch_size): torch.matmul(x,y)
  end.record(); torch.cuda.synchronize()
  row={"label":a.label,"batch":batch,"start_unix_s":f"{stamp:.6f}","operations":a.batch_size,"wall_ms":f"{(time.perf_counter()-wall)*1000:.6f}","cuda_batch_ms":f"{start.elapsed_time(end):.6f}"}
  w.writerow(row); rows.append(row); batch+=1
timings=[float(r["cuda_batch_ms"]) for r in rows]
Path(a.summary).write_text(json.dumps({"label":a.label,"result":"completed","batches":len(rows),"operations":len(rows)*a.batch_size,"matrix_size":a.matrix_size,"batch_size":a.batch_size,"mean_cuda_batch_ms":sum(timings)/len(timings),"pid":os.getpid()},indent=2),encoding="utf-8")
