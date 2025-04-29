import os
import random
import json

# open jsonl file
with open('meta_Toys_and_Games.jsonl', 'r') as f:
    # randomly choose 50 lines
    lines = f.readlines()
    # shuffle the lines
    random.shuffle(lines)

    # take the first 50 lines
    lines = lines[:50]

    # write to a new file
    with open('meta_Toys_and_Games_sample.jsonl', 'w') as f_out:
        for line in lines:
            f_out.write(line)

with open('meta_Cell_Phones_and_Accessories.jsonl', 'r') as f:
    # randomly choose 50 lines
    lines = f.readlines()
    # shuffle the lines
    random.shuffle(lines)

    # take the first 50 lines
    lines = lines[:50]

    # write to a new file
    with open('meta_Cell_Phones_and_Accessories_sample.jsonl', 'w') as f_out:
        for line in lines:
            f_out.write(line)