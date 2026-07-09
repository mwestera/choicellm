"""Turn choice probabilities into result rows, and write them as CSV. Pure (stdlib only)."""
import csv
import io

N_DECIMALS = 6


def _argmax(xs) -> int:
    return max(range(len(xs)), key=lambda i: xs[i])


def _round(probs) -> list[float]:
    return [round(p, N_DECIMALS) for p in probs]


def join(values, delimiter=';') -> str:
    """Join values into a single (properly quoted) delimited string."""
    output = io.StringIO()
    csv.writer(output, delimiter=delimiter).writerow(values)
    return output.getvalue().strip()


def scalar_row(item, probs, scale) -> dict:
    max_index = _argmax(probs)
    probs = _round(probs)
    return {
        'target_id': item.target_id,
        'target': item.target,
        'pred': scale[max_index],
        'prob': probs[max_index],
        'rating': round(sum(p * n for p, n in zip(probs, scale)), N_DECIMALS),
        'probs': join(probs),
    }


def categorical_row(item, probs, category_names) -> dict:
    max_index = _argmax(probs)
    probs = _round(probs)
    return {
        'target_id': item.target_id,
        'target': item.target,
        'pred': category_names[max_index],
        'prob': probs[max_index],
        'probs': join(probs),
    }


def comparative_row(item, probs) -> dict:
    max_index = _argmax(probs)
    probs = _round(probs)
    row = {}
    if item.target_id is not None:
        row['target_id'] = item.target_id
    if item.comparison_id is not None:
        row['comparison_id'] = item.comparison_id
    if item.position is not None:
        row['position'] = item.position
    if item.target is not None:
        row['target'] = item.target
    row['choices'] = join(item.choices)
    row['pred'] = item.choices[max_index]
    if item.position is not None:
        row['prob'] = probs[item.position]
    row['probs'] = join(probs)
    return row


class CsvWriter(csv.DictWriter):
    """A csv.DictWriter that infers its fieldnames from -- and writes the header on -- the first row."""

    def __init__(self, f):
        super().__init__(f, fieldnames=[])

    def writerow(self, rowdict):
        if not self.fieldnames:
            self.fieldnames.extend(rowdict.keys())
            self.writeheader()
        return super().writerow(rowdict)
