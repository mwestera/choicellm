"""Command-line entry point: parse args, build the template + backend, score each item, write CSV."""
import argparse
import logging
import os
import random
import sys

from . import results, sampling
from .backends import make_backend
from .inputs import read_inputs
from .templates import PromptTemplate

DEFAULT_MODEL = "unsloth/Llama-3.2-1B"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        'choicellm',
        description='Rate items on a scale, compare them, or classify them, using LLM choice probabilities.',
    )
    p.add_argument('file', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
                   help='Plaintext file with one item per line. See --csv / --newlines for other formats. '
                        'For comparative mode, a --csv file may also have n_choices columns of predetermined choices.')
    p.add_argument('--prompt', required=True, type=argparse.FileType('r'),
                   help='Prompt template .json file (create one with choicellm-template and adapt it). '
                        'The template also sets the mode: scalar, comparative or categorical.')
    p.add_argument('--model', default=DEFAULT_MODEL,
                   help='A local huggingface model, or a chat model served over an OpenAI-compatible API '
                        '(add --openai or --base-url in that case).')
    p.add_argument('-o', '--output', type=argparse.FileType('w'), default=sys.stdout,
                   help='Where to write the CSV results (default: stdout).')
    p.add_argument('-v', '--verbose', action='store_true', help='Show info logging (device, prompt, per-item output).')

    backend = p.add_argument_group('backend selection')
    backend.add_argument('--openai', action='store_true',
                         help='Use the OpenAI API; otherwise --model is a local huggingface model.')
    backend.add_argument('--base-url', dest='base_url', default=os.environ.get('OPENAI_BASE_URL'),
                         help='Base URL of an OpenAI-compatible server, e.g. http://localhost:8080/v1 for a local '
                              'llama.cpp / vLLM / Ollama server. Implies API mode. Defaults to $OPENAI_BASE_URL.')
    backend.add_argument('--tokenizer',
                         help="Huggingface tokenizer for computing label token ids (logit_bias) against a non-OpenAI "
                              "API server. If omitted for a non-OpenAI model, logit_bias is disabled.")
    backend.add_argument('--dtype', help='Torch dtype for local models, e.g. "auto", "float16", "bfloat16" (default float32).')

    fmt = p.add_argument_group('input format').add_mutually_exclusive_group()
    fmt.add_argument('--csv', action='store_true',
                     help='Input is CSV (one column, or n_choices columns for predetermined comparative choices).')
    fmt.add_argument('--newlines', action='store_true',
                     help='Turn literal "\\n" in the input file into real newlines.')

    comp = p.add_argument_group('comparative mode (when comparisons are not predetermined in a --csv input)')
    comp.add_argument('--compare-to', dest='compare_to', type=argparse.FileType('r'),
                      help='File of items to compare against (default: the input items themselves).')
    comp.add_argument('--compare-deterministic', dest='compare_deterministic', action='store_true',
                      help='Reuse the seed for every item, so each item gets the same set of comparisons.')
    comp.add_argument('--n-comparisons', dest='n_comparisons', type=int, default=100, help='Comparisons per item.')
    comp.add_argument('--all-positions', dest='all_positions', action='store_true',
                      help='Average over all positions of the target item among the choices.')
    comp.add_argument('--seed', type=int, default=None, help='Random seed for comparison sampling (default: random).')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format='%(message)s')

    rng = random.Random()
    if args.seed is None:
        args.seed = rng.randint(0, 99999)
    rng.seed(args.seed)
    logging.info(f'seed = {args.seed}')

    template = PromptTemplate.from_json(args.prompt, rng=rng)
    inputs, is_multicolumn = read_inputs(args.file, args.csv, args.newlines)

    _warn_about_mismatches(args, template, is_multicolumn)

    backend = make_backend(
        args.model,
        template.labels_for_logits,
        use_api=args.openai or bool(args.base_url),
        base_url=args.base_url,
        tokenizer_name=args.tokenizer,
        prompt_start_for_cache=template.prompt_start_for_cache,
        dtype=args.dtype,
    )

    items = _items_for(template, inputs, is_multicolumn, args, rng)

    logging.info(f'-------\n{template}\n-------')

    writer = results.CsvWriter(args.output)
    for item in items:
        probs = backend.probs(template.format(*item.format_args))
        writer.writerow(template.result(item, probs))


def _items_for(template, inputs, is_multicolumn, args, rng):
    if template.mode != 'comparative':
        return sampling.basic_items(inputs)

    if is_multicolumn:
        if args.compare_to:
            logging.warning('WARNING: --compare-to is ignored, because the input file is multi-column csv.')
        return sampling.multicolumn_items(inputs)

    if args.compare_to:
        compare_to, compare_multicolumn = read_inputs(args.compare_to, args.csv, args.newlines)
        if compare_multicolumn:
            raise ValueError('The --compare-to file must not contain multiple .csv columns.')
        compare_to = list(compare_to)
    else:
        inputs = compare_to = list(inputs)

    return sampling.comparison_items(
        inputs, compare_to, template.n_choices, args.n_comparisons, args.all_positions, rng,
        seed_per_item=args.seed if args.compare_deterministic else None,
    )


def _warn_about_mismatches(args, template, is_multicolumn):
    if is_multicolumn and template.mode != 'comparative':
        raise ValueError("Input .csv has multiple columns, which is only valid for 'comparative' mode.")
    if args.model == DEFAULT_MODEL:
        logging.warning(f'WARNING: Using the small default model {args.model}; results may be poor. Use --model to override.')
    if not args.openai and 'gpt' in args.model.lower():
        logging.warning('WARNING: If you meant a model on the OpenAI API, add --openai.')
    if (args.openai or args.base_url) and not template.is_chat:
        logging.warning("WARNING: With an OpenAI-compatible API you're advised to use a chat-style prompt.")
    if 'instruct' in args.model.lower() and not template.is_chat:
        logging.warning('WARNING: With an "instruct" model you\'re advised to use a chat-style prompt.')


if __name__ == '__main__':
    main()
