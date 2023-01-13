# Minsight-sensor

By Iris Andrussow, Huanbo Sun, Katherine J. Kuchenbecker, Georg Martius

## Table of Contents

0. [Introduction](#introduction)
0. [Requirements](#requirements)
0. [Usage](#usage)

## Introduction

This repository contains PyTorch implementation code for the paper "Minsight: a fingertip-sized vision-based tactile sensor for robotic manipulation", currently under review. This work introduces a machine-learning based fingertip-sized vision-based haptic sensor. Experimental data can be found under https://doi.org/10.17617/3.AEDHD1 

## Requirements

PyTorch >= 1.9.0

`pip install -r requirements.txt`

## Usage

Data processing for the raw data can be found in the jupyter notebook under data_processing/. \
Code for training the mapping on the preprocessed data, can be found in training/ and run with `python main.py` \
To reproduce error plots and inference times, refer to code in experiments/. \

To reproduce results for the lump classification experiment, use code in lump_classification/, together with the data published in https://doi.org/10.17617/3.AEDHD1

