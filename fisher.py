import random
import time
import argparse
import numpy as np


def gen_line(init, target, amp=0.05, inter=1000):
    x = np.linspace(0, 100, inter)
    k = (target - init) / 100
    b = np.random.random(size=(inter)) * amp
    y = k * x + init + b
    return x, y


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--iters', type=int, default=20)
parser.add_argument('-mi', '--max_iter', type=int, default=400)
parser.add_argument('-ir', '--init_r', type=np.float64, default=0.2)
parser.add_argument('-tr', '--target_r', type=np.float64, default=0.5)
parser.add_argument('-ar', '--amp_r', type=np.float64, default=0.05)
parser.add_argument('-ip', '--init_p', type=np.float64, default=0.3)
parser.add_argument('-tp', '--target_p', type=np.float64, default=0.7)
parser.add_argument('-ap', '--amp_p', type=np.float64, default=0.05)
parser.add_argument('-il', '--init_l', type=np.float64, default=0.8)
parser.add_argument('-tl', '--target_l', type=np.float64, default=0.1)
parser.add_argument('-al', '--amp_l', type=np.float64, default=0.05)
args = parser.parse_args()

print("Load Training Data...")
time.sleep(2)
print("Dataset Done. Shape:(100000,128,128,3)")
time.sleep(1)
print("Super wxm_cnn pretrained model Done.")
print(
    "wxm_cnn(batch_size=16, crop_h=64, crop_w=64, depth=32, max_iter=100, lr=0.15, momentum=0.9, optimizer='sgd', weight_decay=0.0005, shuffle=True, verbose=True, rangdom_state=10, logs_dir='~/Document/fake/logs, workers=12)\n"
)
time.sleep(2)

epc = 0
count = 0

recall_init_x, recall_init_y = gen_line(init=args.init_r,
                                        target=args.target_r,
                                        amp=args.amp_r,
                                        inter=args.max_iter)
prec_init_x, prec_init_y = gen_line(init=args.init_p,
                                    target=args.target_p,
                                    amp=args.amp_p,
                                    inter=args.max_iter)
loss_init_x, loss_init_y = gen_line(init=args.init_l,
                                    target=args.target_l,
                                    amp=args.amp_l,
                                    inter=args.max_iter)

while (True):

    time.sleep(1)
    print("Epoch:{}\n".format(epc))
    time.sleep(2)
    for inter in range(args.iters):

        if count < len(recall_init_y):
            recall = recall_init_y[count]
        else:
            recall = target_r + random.random() * 0.05

        if count < len(prec_init_y):
            prec = prec_init_y[count]
        else:
            prec = target_p + random.random() * 0.05

        if count < len(loss_init_y):
            los = loss_init_y[count]
        else:
            los = target_l + random.random() * 0.05

        cost = random.random()
        print(
            "Interation:{}  Recall:{},  Precision:{},  Loss:{},  Time:{:.2f}s".
            format(inter, recall, prec, los, cost))
        count += 1
        time.sleep(1)
    time.sleep(1)
    print("\n * Finished Epoch {}  Top 5 Prec {:.2f}% 	Top 5 Recall {:.2f}%\n".
          format(epc, prec * 100, recall * 100))
    epc += 1
