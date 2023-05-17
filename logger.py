import functools
import logging
import os
import sys
import torch
from typing import Optional

from termcolor import colored

__all__ = ["setup_logger", "get_logger"]


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def setup_logger(
    output: Optional[str] = None, distributed_rank: int = 0, *, mode: str = 'w',
    color: bool = True, name: str = "exp", abbrev_name: Optional[str] = None
):
    """Initialize the graphwar logger and set its verbosity level to "DEBUG".

    Parameters
    ----------
    output : Optional[str], optional
        a file name or a directory to save log. If None, will not save log file.
        If ends with ".txt" or ".log", assumed to be a file name.
        Otherwise, logs will be saved to `output/log.txt`.
    distributed_rank : int, optional
        used for distributed training, by default 0
    mode : str, optional
        mode for the output file (if output is given), by default 'w'.
    color : bool, optional
        whether to use color when printing, by default True
    name : str, optional
        the root module name of this logger, by default "graphwar"
    abbrev_name : Optional[str], optional
        an abbreviation of the module, to avoid long names in logs.
        Set to "" to not log the root module in logs.
        By default, None.

    Returns
    -------
    logging.Logger
        a logger

    Example
    -------
    >>> logger = setup_logger(name='my exp')

    >>> logger.info('message')
    [12/19 17:01:43 my exp]: message

    >>> logger.error('message')
    ERROR [12/19 17:02:22 my exp]: message

    >>> logger.warning('message')
    WARNING [12/19 17:02:32 my exp]: message

    >>> # specify output files
    >>> logger = setup_logger(output='log.txt', name='my exp')
    # additive, by default mode='w' 
    >>> logger = setup_logger(output='log.txt', name='my exp', mode='a')    

    # once you logger is set, you can call it by
    >>> logger = get_logger(name='my exp')
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if abbrev_name is None:
        abbrev_name = name

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    # stdout logging: master only
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
                abbrev_name=str(abbrev_name),
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")

        if distributed_rank > 0:
            filename = filename + ".rank{}".format(distributed_rank)

        dirs = os.path.dirname(filename)
        if dirs:
            if not os.path.isdir(dirs):
                os.makedirs(dirs)
        file_handle = logging.FileHandler(filename=filename, mode=mode)
        file_handle.setLevel(logging.DEBUG)
        file_handle.setFormatter(plain_formatter)
        logger.addHandler(file_handle)

    return logger


def get_logger(name: str = "GraphWar"):
    """Get a logger for a given name.

    Parameters
    ----------
    name : str, optional
        name of the logger, by default "GraphWar"

    Returns
    -------
    a logger for the given name
    """
    return logging.getLogger(name)


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


class Statistics(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 2
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None, f=sys.stdout, last_best=False):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            if last_best:
                # get last max value index by reversing result tensor
                argmax = result.size(0) - result[:, 0].flip(dims=[0]).argmax().item() - 1
            else:
                argmax = result[:, 0].argmax().item()
            print(f'Run {run + 1:02d}:', file=f)
            print(f'Highest Valid: {result[:, 0].max():.2f}', file=f)
            print(f'Highest Eval Point: {argmax + 1}', file=f)
            print(f'   Final Test: {result[argmax, 1]:.2f}', file=f)
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []

            for r in result:
                valid = r[:, 0].max().item()
                if last_best:
                    # get last max value index by reversing result tensor
                    argmax = r.size(0) - r[:, 0].flip(dims=[0]).argmax().item() - 1
                else:
                    argmax = r[:, 0].argmax().item()
                test = r[argmax, 1].item()
                best_results.append((valid, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:', file=f)
            r = best_result[:, 0]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}', file=f)
            r = best_result[:, 1]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}', file=f)
            return r.mean().cpu().item(), r.std().cpu().item()