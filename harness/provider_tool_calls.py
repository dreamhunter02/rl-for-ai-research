"""Parse explicit provider tool-call envelopes; never execute generated text."""
from __future__ import annotations

import json
import re

MAX_TEXT_CHARS = 200000
MAX_CALLS = 50
NAME = r'[A-Za-z_][A-Za-z0-9_]*'
BLOCK = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL)
FUNCTION = re.compile(r'<function=(' + NAME + r')>\s*(.*?)\s*</function>', re.DOTALL)
PARAMETER = re.compile(r'<parameter=(' + NAME + r')>\s*(.*?)\s*</parameter>', re.DOTALL)


def parse_provider_tool_calls(text, start_id=0):
    """Recover Qwen XML or JSON calls only within explicit tool_call markers.

    Unknown names remain attempted calls for ordinary runtime rejection. JSON
    parameter types are preserved; unquoted string parameters stay strings.
    Malformed or ambiguous envelopes are ignored rather than guessed from prose.
    """
    if not isinstance(text, str) or len(text) > MAX_TEXT_CHARS:
        return []
    calls = []
    for block in BLOCK.finditer(text):
        body = block.group(1).strip()
        name, arguments = None, None
        function = FUNCTION.fullmatch(body)
        if function:
            name, parameters = function.groups()
            arguments = {}
            consumed = 0
            valid = True
            for parameter in PARAMETER.finditer(parameters):
                if parameters[consumed:parameter.start()].strip():
                    valid = False
                    break
                key, value = parameter.groups()
                if key in arguments:
                    valid = False
                    break
                try:
                    arguments[key] = json.loads(value)
                except json.JSONDecodeError:
                    arguments[key] = value.strip()
                consumed = parameter.end()
            if not valid or parameters[consumed:].strip():
                continue
        else:
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            name = payload.get('name')
            arguments = payload.get('arguments')
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    continue
        if not isinstance(name, str) or not re.fullmatch(NAME, name) or not isinstance(arguments, dict):
            continue
        calls.append({'id': f'provider-text-call-{start_id + len(calls)}', 'type': 'function',
                      'function': {'name': name, 'arguments': json.dumps(arguments, ensure_ascii=False)}})
        if len(calls) >= MAX_CALLS:
            break
    return calls
