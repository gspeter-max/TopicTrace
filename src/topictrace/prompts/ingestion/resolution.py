SYSTEM_PROMPT = """Analyze the list of entity names and find pairs that refer to the same real-world entity (e.g. person, company, place).

For any pair of matching names:
- Identify the best, most complete, and most correct version as `canonical_name`.
- Output a decision with `left_name` and `right_name` set to the two matching names.

Only output decisions for names that should be merged. If names do not match or are not similar, do not include them in the output.

Return only a JSON object with a `decisions` array.
Each item in `decisions` must look like this:
{
  "left_name": "...",
  "right_name": "...",
  "canonical_name": "..."
}"""

USER_PROMPT_TEMPLATE = """Here is the list of entity names to check:

{payload_json}

Return only a JSON object with a `decisions` array."""