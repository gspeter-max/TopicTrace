SYSTEM_PROMPT = """Look at pairs of names and decide whether they refer to the same real-world thing, such as a person, company, or place.

Merge only when you are completely confident they are the same thing. If the names differ only in small formatting changes, punctuation changes, abbreviations, or common naming variations, treat them as the same only when that is clearly correct.

If two names are the same:
- set `should_merge` to true
- set `canonical_name` to the best, most complete, and most correct version of the name

If two names are not the same, or you are not sure:
- set `should_merge` to false
- leave `canonical_name` empty

Return only a JSON object with a `decisions` array.
Each item in `decisions` must look like this:
{
  "left_name": "...",
  "right_name": "...",
  "should_merge": true or false,
  "canonical_name": "..."
}"""

USER_PROMPT_TEMPLATE = """Here are the name pairs to check.

{payload_json}

Return only a JSON object with a `decisions` array."""