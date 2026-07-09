"""Reading the input file of items. Pure (stdlib only)."""
import csv
import itertools
from typing import Iterator, Union


def read_inputs(file, is_csv=False, escaped_newlines=False) -> tuple[Iterator[Union[str, list[str]]], bool]:
    """
    Yield one item per input line. Returns (items, is_multicolumn).

    - default: one plain item per line (stripped).
    - escaped_newlines: as above, but literal '\\n' is turned into a real newline.
    - is_csv: parse as CSV. A single column behaves like the plain case (but allows quoting/newlines);
      multiple columns are the predetermined choices for comparative mode (is_multicolumn=True).
    """
    if is_csv:
        rows = csv.reader(file)
        first_row = next(rows)
        if len(first_row) > 1:
            return itertools.chain([first_row], rows), True
        return itertools.chain([first_row[0]], (row[0] for row in rows)), False
    if escaped_newlines:
        return (line.strip().replace('\\n', '\n') for line in file), False
    return (line.strip() for line in file), False
