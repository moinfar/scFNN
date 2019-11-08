import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Transpose a csv file')
parser.add_argument('-i', '--input', metavar='INPUT', type=str, required=True)
parser.add_argument('-o', '--output', metavar='INPUT', type=str, required=True)

args = parser.parse_args()

data = pd.read_csv(args.input, index_col=0)
data.transpose().to_csv(args.output)
