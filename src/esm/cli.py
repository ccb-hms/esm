import os
import argparse
import logging
from pathlib import Path
from esm.evoprotgrad import EvoProtGrad
from esm.evoprotgrad import torch_device

testsequence = 'MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN'

parser = argparse.ArgumentParser(description='ESM2 command line inference')
parser.add_argument('-a', '--app', default='evolution', required=True, help='module name')
parser.add_argument('-s', '--sequence', default=testsequence, help='protein sequence')
parser.add_argument('-o', '--output_file', default='output.csv', help='path to output csv file')

app_list = ['evolution']


def main(args=None):
    args = parser.parse_args(args)
    output_file = args.output_file
    output_directory = os.path.dirname(output_file)
    if not os.path.exists(output_directory):
        output_directory = os.path.join(os.environ.get('HOME'), args.app)
        output_file = os.path.join(output_directory, os.path.basename(args.output_file))
        Path(output_directory).mkdir(parents=True, exist_ok=True)

    if args.app not in app_list:
        print(f'Invalid app name {args.app}')

    else:
        log_file = os.path.join(output_directory, f'{args.app}.log')
        dtfmt = '%y%m%d-%H:%M'
        logfmt = '%(asctime)s-%(name)s-%(levelname)s-%(message)s'
        logging.basicConfig(filename=log_file,
                            filemode='w',
                            level=logging.INFO,
                            format=logfmt,
                            datefmt=dtfmt)
        logger = logging.getLogger(name=__name__)

        if args.app == 'evolution':
            print(f'Single protein evolution on target protein:\n{args.sequence}')
            epg = EvoProtGrad()
            device_dict = torch_device()
            print(device_dict)
            output_df = epg.single_evolute(raw_protein_sequence=args.sequence)
            try:
                output_df.to_csv(output_file, index=False)
            except Exception as e:
                logger.warning(f'Cannot write output file {output_file}:\n{e}')
