"""
A parser for FASTA files.

It can handle files that are local or on the web.
Gzipped files do not need to be unzipped.
"""

import os
from urllib.request import urlopen


def myopen(fileName):
    if not (os.path.exists(fileName) and os.path.isfile(fileName)):
        raise ValueError('file does not exist at %s' % fileName)

    import gzip
    fileHandle = gzip.GzipFile(fileName)

    gzippedFile = True
    try:
        line = fileHandle.readline()
        fileHandle.close()
    except:
        gzippedFile = False

    if gzippedFile:
        return gzip.GzipFile(fileName)
    else:
        return open(fileName)


class MalformedInput:
    "Exception raised when the input file does not look like a fasta file."
    pass


class FastaRecord:
    """Represents a record in a fasta file."""

    def __init__(self, header, sequence):
        """Create a record with the given header and sequence."""
        self.header = header
        self.sequence = sequence

    def __str__(self):
        return '>' + self.header + '\n' + self.sequence + '\n'


def _fasta_itr_from_file(file_handle):
    "Provide an iteration through the fasta records in file."

    h = file_handle.readline()[:-1]
    if h[0] != '>':
        raise MalformedInput()
    h = h[1:]

    seq = []
    for line in file_handle:
        line = line[:-1]  # remove newline
        if line:
            if line[0] == '>':
                yield FastaRecord(h, ''.join(seq))
                h = line[1:]
                seq = []
                continue
            seq.append(line)

    yield FastaRecord(h, ''.join(seq))


def _fasta_itr_from_web(file_handle):
    "Iterate through a fasta file posted on the web."

    h = file_handle.readline().decode("utf-8")[:-1]
    if h[0] != '>':
        raise MalformedInput()
    h = h[1:]

    seq = []
    for line in file_handle:
        line = line.decode("utf-8")[:-1]  # remove newline
        if line[0] == '>':
            yield FastaRecord(h, ''.join(seq))
            h = line[1:]
            seq = []
            continue
        seq.append(line)

    yield FastaRecord(h, ''.join(seq))


def _fasta_itr_from_name(fname):
    "Iterate through a fasta file with the given name."

    f = myopen(fname)
    for rec in _fasta_itr_from_file(f):
        yield rec


def _fasta_itr(src):
    """Provide an iteration through the fasta records in file `src'.

    Here `src' can be either a file name or a url of a file.
    """
    if type(src) == str:
        if src.find("http") >= 0:
            file_handle = urlopen(src)
            return _fasta_itr_from_web(file_handle)
        else:
            return _fasta_itr_from_name(src)
    else:
        raise TypeError


class fasta_itr(object):
    """An iterator through a Fasta file"""

    def __init__(self, src):
        """Create an iterator through the records in src."""
        self.__itr = _fasta_itr(src)

    def __iter__(self):
        return self

    def __next__(self):
        return self.__itr.__next__()
