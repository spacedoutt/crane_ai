### Polygon Data Analysis

> **About Myself**: I'm an aspiring ML/AI Engineer using this repo to help teach myself model training and to become more advanced in AI practices.

## Overview

This repo has a few important parts. First are the folders, these all store a different form of trained AI. I will start adding the metrics for those training sets into the these model folders, this change will occur on model_4+. The python files are made for either training or obtaining polygon news data. The polygon.csv file is the stored news data from Feb, 23, 2024 at 20:45:06 going forward, it will auto update each time I run the file.

polygon.csv dataframe information:

- **Title:** Article title.
- **Content:** Contents of the entire article scraped from the link.
- **Link:** Link to the article.
- **Date:** Date the article was posted

## Source

This is my own work that I have derived from my own learning on AI and many youtube videos, informational websites, and assisted programming tools.

## Usage

1. Download the repo
2. Run py main.py --type {chat, train} --model_name {model name} --compare-trained-model {True, False}

## main.py

The main python file here has a few important tag functionalities.

- **type** is used to determine what the program should be looking to do. If you want to use or chat with the model in question run the "chat" type. If you want to train a new model run the "train" type. Default value is train
- **model_name** is used to determine what model you want to run, wether it be
