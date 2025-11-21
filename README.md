# ChoiceLLM: A Python package for ratings and multiple choice answers from LLMs 

## Purpose

This tool makes it easier to prompt an LLM for:

- 'scalar' judgments (e.g., "On a scale from 1 to 7, how abstract is this word?", "How suggestive is this question?").
- 'comparative' judgments (e.g., "Of these 4 words, which is the most emotional?")
- 'categorical' judgments, i.e., to classify words, phrases or paragraphs into a number of fixed, predetermined categories (e.g., "What type of speech act is this utterance?").

and to do so for a large amount of 'stimuli', and using different LLMs (both locally running LLMs, and models via the OpenAI API, if you have an `OPENAI_API_KEY`).

The scores obtained are deterministic, obtained by accessing the underlying logits computed by the model (as opposed to sampling a concrete response).

## Installation

From PyPI:

```bash
pip install choicellm
```

Installation makes available the main program `choicellm`, and helper programs `choicellm-template` and `choicellm-aggregate`.

## First create a suitable prompt JSON file (`choicellm-template`)

When running `choicellm`, you need to provide it with a 'prompt template' file in JSON format, like `choicellm --prompt my_first_prompt.json`. The exact format of this file will depend on what you want to use `choicellm` for, i.e., for scalar judgments (and if so, on what kind of scale), for multiple-choice questions, or for comparing items, and whether you plan to use a 'base' model (i.e., plain language generation) or an 'instruct' model (i.e., chatbot like GPT).

To generate an example prompt template (which you can then modify to suit your needs), you can do, for instance:

```bash
choicellm-template --scalar > my_first_prompt.json
```

Alternative modes are `--comparative` and `--categorical` (see below for some explanation, but you'll understand most from the generated prompt templates themselves). If you want to use an 'instruct' model, additionally specify `--chat`. (Note that the examples were written for a study on concreteness vs. abstractness; you can of course modify all of this.)

When modifying a prompt, the parts in curly braces, like `{scale}`, will be replaced by the program, according to the information you provide. You may also omit such parts (by removing `{scale}`), for instance if you think they are unnecessary or you manually include the information in the prompt.

The prompt templates contain some additional comments under the JSON key `_comments`.

## Basic usage

Now, assuming you have some items (words, phrases) to be categorized (or whatever) in a plain text file `items.txt`, one on each line, you can now do:

```bash
choicellm items.txt --prompt my_first_prompt.json
```

Results will simply be printed to the command-line (along with some logging info). To save them, use output redirection:

```bash
choicellm items.txt --prompt my_first_prompt.json > results.csv
```

The default model won't be very big, or very good. If you have a more powerful computer, you could try something larger with the `--model` option, accepting any model identifier from the Huggingface hub.

```bash
choicellm items.txt --model 'unsloth/llama-3-70b-bnb-4bit' --prompt my_first_prompt.json > results.csv
```

To use a proprietary OpenAI model, first, you need to set the `OPENAI_API_KEY` environment variable to your secret key:

```bash
export OPENAI_API_KEY=yoursecretkey123
```

Then you can specify an OpenAI model (and include the `--openai` flag):

```bash
choicellm items.txt --model 'gpt-4o' --openai --my_first_prompt.json > results.csv
```

Since GPT-4o is an 'instruct' (i.e., chat) model, make sure you use it with a prompt template created with the `--chat` option, e.g., `choicellm-template --scalar --chat`.

On the whole, while `choicellm` is probably not entirely model-agnostic in some unforeseen ways, it should work with Huggingface models and OpenAI models. Reasonable local options that fit in a 48GB GPU: `unsloth/llama-3-70b-bnb-4bit` or `unsloth/Llama-3.3-70B-Instruct-bnb-4bit`. Since the latter is an 'instruct' model, again make sure to use a prompt template generated with the `--chat` option.

## A bit more info about the three 'modes': scalar, comparative, categorical

First, `--scalar` prompts ask the LLM to give ratings on a numerical scale. Beware that negative scale values (like a scale -1, 0, 1, which could make sense for sentiment) are currently supported only for local models with `transformers`, not through the OpenAI client. (This applies more generally to labels that do not map onto single token ids for the given tokenizer.) The `results.csv` file contains the items, with additional columns `pred` (the model's choice), `prob` (the probability the model assigned to this choice, i.e., the softmaxed logits), `rating` (a weighted sum taking the model's probabilities for all choices into account), `probs` (the probabilities for all choices, separated by semicolons; these sum to 1.0). 

Next, `--categorical` prompts are most suited for single-label categorization: the category probabilities sum to one (under the hood, the LLM is not given the option of listing several categories). (For multi-label categorization, see below.) For this mode, the `results.csv` file contains `pred` (the 'chosen' category), `prob` (its probability) and `probs` (probabilities for all categories, separated by semicolons).

Third, `--comparative` prompts ask the model to compare each item to a large number of random other items, which can result in more reliable per-item scores (once aggregated over all comparisons). Applying `choicellm` to such a prompt comes with some additional options, see `choicellm --help` for that. (Publication with more explanation pending.) Note that, for a `--comparative` prompt, the `results.csv` file will not contain a single score per item, but rather a separate row for each of the many comparisons it did. The auxiliary command `choicellm-aggregate` may be used to aggregate these comparisons into a single score per item (in this case resulting in values on the scale 1,5):

```bash
choicellm-aggregate results.csv --scale 1,5 > results_aggregated.csv
```

Lastly, to achieve _multi-label_ classification (where the same instance may fit several categories at once), the recommended approach is to implement this as multiple, separate `--scalar` (or `--comparative`) prompts, one for each category, and fed separately into `choicellm`. After creating separate prompt files (each named as `<category>.json`), feeding them into `choicellm` can be easily done in Bash by looping over the prompt files as follows (e.g.):

```bash
for category in games movies music sports; do
    choicellm items.txt --model 'unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit' --prompt "$category.json" > "$category.csv"
done
```

Having a separate prompt template per category (with designated system prompt and few-shot examples for each category) is probably a good idea regardless, as it encourages fine-tuning and evaluating the system for each category separately. The `.csv` results files (one per category) need to be merged afterward, which is fairly straightforward using (for instance) the `pandas` library. 

