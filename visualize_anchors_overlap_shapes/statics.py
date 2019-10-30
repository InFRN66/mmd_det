"""
take statics result from csv files (random sample, and get positives)
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
import csv
import pandas as pd
import argparse


def to_list01(argument):
    def f(x): return x.split(",")
    return list(map(float, f(argument)))


parser = argparse.ArgumentParser(description="options")
parser.add_argument("--sep", default=4, type=int,
                    help="number of separation e.g. sep=4, 0-75-150-225-300")
parser.add_argument("--file", type=str, help="objective file path")
parser.add_argument('--thresholds', type=to_list01,
                    help='various threshold', default=None)
args = parser.parse_args()


def overlap_all(data):
    num_objects = len(data)
    all_num_anchors = int(data[["num_anchors"]].sum())
    try:
        out = all_num_anchors / num_objects
    except ZeroDivisionError:
        out = 0
    return out


def overlap_accumurate_area(data, points=10, max_res=300):
    criterion = (np.sqrt(data[["gt_size"]])*max_res).values  # --- 300*√hw
    x = np.linspace(0, max_res, points+1)
    result = list()
    for thresh in x:
        mask_data = data[criterion <= thresh]
        result.append(overlap_all(mask_data))
    return x, result


def overlap_per_range(data, points=6, max_res=300):
    data = data.copy()
    criterion = (np.sqrt(data[["gt_size"]])*max_res).values
    data["criterion"] = criterion
    x = np.linspace(0, max_res, points+1)
    bins = list()
    for i in range(len(x)-1):
        mask_data = data.values[np.where(
            (criterion >= x[i]) & (criterion < x[i+1]))[0]]
        mask_data = pd.DataFrame(mask_data, columns=data.columns)
        bins.append(overlap_all(mask_data))
    return x, bins


def show_plot(x, y):
    plt.rcParams["font.size"] = 16
    plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    plt.grid(True)
    plt.ylabel("mean positive anchors")
    plt.xlabel("ground truth box scale")
    plt.title("pos anchors lt")
    plt.show()


def show_bar(x, y, save=False):
    plt.rcParams["font.size"] = 16
    plt.figure(figsize=(8, 6))
    plt.bar(x[:-1], height=y, width=x[1], align="edge")
    plt.grid(True)
    plt.ylabel("mean positive anchors")  # dfasdfsdf
    plt.xlabel("ground truth box scale")  # fsdfsadf
    # plt.xticks(x)
    plt.title("pos anchors per range")
    if save:
        plt.savefig(
            f"anchor_overlap_statics/{args.file.split('_')[-1]}_{args.sep}sep.png")
    # plt.show()


def show_bar_thresh(X, Y, save=False):
    plt.rcParams["font.size"] = 16
    plt.figure(figsize=(10, 6))
    plt.rcParams['legend.fontsize'] = 13
    plt.rcParams['legend.handlelength'] = 2
    for i, key in enumerate(X.keys()):
        if i == 0:
            min_ = key
        x = X[key]
        y = Y[key]
        plt.bar(x[:-1], height=y, width=x[1], align="edge", label=key)
        plt.grid()
        plt.ylabel("mean positive anchors")
        plt.xlabel("ground truth box scale")
        plt.title("pos anchors per range")
    max_ = key
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0)
    plt.subplots_adjust(left=0.1, right=0.8)
    if save:
        plt.savefig(f"anchor_overlap_statics/{min_}-{max_}_{args.sep}sep.png")
    plt.show()


if __name__ == "__main__":
    # print("All Mean anchors: ", overlap_all(data))
    # print()
    if args.thresholds is None:
        with open(args.file, "r") as f:
            data = pd.read_csv(f)
        print(f"load {args.file}!")
        x, y = overlap_per_range(data, points=args.sep)
        show_bar(x, y, save=True)
    else:
        X, Y = dict(), dict()
        for thresh in args.thresholds:
            with open(f"./stat_csv/statics_anchors1234_thresh{thresh}.csv", "r") as f:
                out = pd.read_csv(f)
            x, y = overlap_per_range(out, points=100)
            X[thresh] = x
            Y[thresh] = y
        show_bar_thresh(X, Y, save=True)
