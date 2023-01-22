import argparse
from argparse import ArgumentParser

import pkg_resources

from od.config.config import Config
from od import __version__


def main():
    parser = ArgumentParser(
        prog=pkg_resources.get_distribution('od').project_name,
        description='Object Detection gRPC service',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-v', '--version',
        action='version',
        version=f'%(prog)s {__version__}',
    )
    subcommands = parser.add_subparsers(
        title='subcommands',
        description='valid subcommands',
        help='additional help',
        dest='subcommand',
    )
    run_parser = subcommands.add_parser(
        'run',
        help='Run the gRPC server',
    )
    run_parser.add_argument(
        '-c', '--config',
        help='Path to the config yaml file',
    )
    args = parser.parse_args()
    if args.subcommand == 'run':
        config = Config(config_file=args.config)
        if config.config_file is not None:
            from od.config.reader import read
            config = read(config)
        from od.server import serve
        serve(config)
    else:
        parser.print_help()
    return None
