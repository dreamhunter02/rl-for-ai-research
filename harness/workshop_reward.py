"""Versioned, SDK-independent workshop evaluator; targets never enter prompts."""
from __future__ import annotations
import ast
import copy
import json
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext

SCORER_VERSION = "workshop-v1"
SCALES = {"ones": Decimal(1), "thousand": Decimal(1000), "million": Decimal(1000000), "billion": Decimal(1000000000)}
SCALE_ALIASES = {"": "ones", "1": "ones", "k": "thousand", "thousands": "thousand", "m": "million", "mm": "million", "millions": "million", "b": "billion", "bn": "billion", "billions": "billion"}
UNIT_ALIASES = {"$": "USD", "usd": "USD", "dollars": "USD", "€": "EUR", "eur": "EUR", "£": "GBP", "gbp": "GBP", "%": "percent", "percentage": "percent", "percent": "percent", "ratio": "ratio", "": "number", "number": "number"}
NUMBER = re.compile(r"^\s*(?P<currency>USD|EUR|GBP|[$€£])?\s*(?P<open>\()?\s*(?P<value>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[+-]?\.\d+)\s*(?P<close>\))?\s*(?P<scale>thousands?|millions?|billions?|bn|mm|[kmb])?\s*(?P<unit>%|percent|ratio|x|USD|EUR|GBP)?\s*$", re.I)


def scalar(value):
    if isinstance(value, bool) or not re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)", str(value).strip()):
        raise ValueError("value must be one finite decimal string, without units or alternatives")
    result = Decimal(str(value).strip())
    if not result.is_finite() or abs(result) > Decimal("1e30"):
        raise ValueError("value outside supported finite range")
    return result


def normalize_unit(value):
    value = str(value).strip()
    return UNIT_ALIASES.get(value.lower(), UNIT_ALIASES.get(value, value.lower()))


def normalize_scale(value):
    value = str(value).strip().lower()
    value = SCALE_ALIASES.get(value, value)
    if value not in SCALES:
        raise ValueError("scale must be ones, thousand, million or billion")
    return value


def parse_number(text):
    match = NUMBER.fullmatch(str(text).replace("−", "-").replace("–", "-"))
    if not match or bool(match['open']) != bool(match['close']):
        raise ValueError("not a single financial number")
    value = scalar(match['value'].replace(',', ''))
    if match['open']:
        value = -abs(value)
    unit = normalize_unit(match['currency'] or match['unit'] or '')
    scale = normalize_scale(match['scale'] or '')
    if match['currency'] and match['unit'] and normalize_unit(match['unit']) != unit:
        raise ValueError("conflicting currency and unit")
    return value, unit, scale, max(0, -value.as_tuple().exponent)


def typed_number(value, unit='number', scale='ones'):
    return scalar(value), normalize_unit(unit), normalize_scale(scale)


def numbers_equal(gold, candidate, precision):
    gv, gu, gs = gold
    cv, cu, cs = candidate
    if gu != cu or (gv < 0) != (cv < 0):
        return False
    with localcontext() as context:
        context.prec = 80  # bounded 1e30 values, scale conversion and up to 12 decimals
        converted = cv * SCALES[cs] / SCALES[gs]
        quantum = Decimal(1).scaleb(-int(precision))
        try:
            return converted.quantize(quantum, rounding=ROUND_HALF_UP) == gv.quantize(quantum, rounding=ROUND_HALF_UP)
        except InvalidOperation:
            return False


def numeric_equal(gold, candidate):
    try:
        gv, gu, gs, precision = parse_number(gold)
        cv, cu, cs, _ = parse_number(candidate)
        return numbers_equal((gv, gu, gs), (cv, cu, cs), precision)
    except (ValueError, InvalidOperation):
        return False


def validate_target(target):
    if target.get('reviewed') is not True:
        raise ValueError('Target must be source-reviewed before training/evaluation')
    kind = target.get('answer_type')
    if kind == 'numeric':
        typed_number(target['value'], target['unit'], target['scale'])
        if target.get('derived'):
            if not target.get('expression') or not target.get('operands'):
                raise ValueError('Derived target requires a reviewed expression and operand specifications')
            for operand in target['operands'].values():
                typed_number(operand['value'], operand['unit'], operand['scale'])
                if not operand.get('support'):
                    raise ValueError('Derived target operands require reviewed supporting spans')
                if not operand.get('metric') or not operand.get('period'):
                    raise ValueError('Derived target operands require metric and period')
        precision = target.get('precision')
        if isinstance(precision, bool) or not isinstance(precision, int) or not 0 <= precision <= 12:
            raise ValueError('Target precision must be an integer 0..12')
    elif kind == 'boolean':
        if target.get('decision') not in ('yes', 'no'):
            raise ValueError('Boolean target requires yes/no')
    elif kind == 'text':
        if not target.get('aliases') and not target.get('required_facts'):
            raise ValueError('Text target requires reviewed aliases or required_facts')
    else:
        raise ValueError('Use numeric/boolean/text; encode multipart answers as reviewed text facts')
    return target


def validate_submission(submission):
    if not isinstance(submission, dict):
        raise ValueError('Submission must be an object')
    result = copy.deepcopy(submission)
    for field_name in ('answer_text', 'calc_id'):
        if field_name in result and not isinstance(result[field_name], str):
            raise ValueError(f'{field_name} must be a string')
    kind = result.get('answer_type')
    if kind == 'numeric':
        value, unit, scale = typed_number(result.get('value', ''), result.get('unit', ''), result.get('scale', ''))
        if not result.get('unit') or not result.get('scale'):
            raise ValueError('Numeric finish requires value, unit and scale')
        # Scalar outputs carry no potentially contradictory prose; justification is evidence/calc.
        if result.get('answer_text', '').strip():
            raise ValueError('Numeric finish must leave answer_text empty; use citations and calc_id')
        result.update(value=str(value), unit=unit, scale=scale)
    elif kind == 'boolean':
        if result.get('decision') not in ('yes', 'no'):
            raise ValueError('Boolean finish requires decision yes/no')
        if result.get('answer_text', '').strip():
            raise ValueError('Boolean finish must leave answer_text empty; use text for multipart claims')
    elif kind == 'text':
        if not str(result.get('answer_text', '')).strip():
            raise ValueError('Text finish requires nonempty answer_text')
    else:
        raise ValueError('answer_type must be numeric, boolean or text')
    citations = result.setdefault('citations', [])
    if not isinstance(citations, list):
        raise ValueError('citations must be a list')
    for c in citations:
        if (not isinstance(c, dict)
                or any(not isinstance(c.get(k), str) or not c[k].strip() for k in ('receipt_id', 'document_id'))
                or type(c.get('page')) is not int or c['page'] < 1):
            raise ValueError('Each citation requires receipt_id, document_id and one-based page')
    return result


@dataclass
class EpisodeState:
    receipts: dict = field(default_factory=dict)
    calculations: dict = field(default_factory=dict)
    accepted: dict | None = None
    turn: int = 0
    validation_errors: int = 0
    terminal_ambiguous: bool = False
    observations: list = field(default_factory=list)
    last_score: dict = field(default_factory=dict)

    def record(self, tool, payload):
        key = f'r{len(self.receipts) + 1}'
        self.receipts[key] = {**copy.deepcopy(payload), 'tool': tool, 'receipt_id': key, 'delivered_turn': self.turn}
        return key

    def calculate(self, expression, operands):
        if len(expression) > 500 or len(operands) > 12:
            raise ValueError('Calculation too large')
        tree = ast.parse(expression, mode='eval')
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        if names != set(operands):
            raise ValueError('Provide exactly one source-backed operand for every expression variable')
        values = {}
        for name, operand in operands.items():
            receipt = self.receipts.get(operand.get('receipt_id'), {})
            quote = operand.get('quote', '')
            if receipt.get('delivered_turn', -1) >= self.turn or receipt.get('tool') not in ('read', 'read_table', 'grep_document') or not quote or quote not in receipt.get('text', ''):
                raise ValueError('Operand quote was not delivered by a strong read')
            if not operand.get('metric') or str(operand['metric']).lower() not in quote.lower() or not operand.get('period') or str(operand['period']) not in quote:
                raise ValueError('Operand metric and period must identify text in its source quote')
            value = scalar(operand['value'])
            # Source values are in displayed units; expression performs explicit conversions.
            mentions = re.findall(r'(?<![\w.])(?:\(\s*[+-]?\d[\d,]*(?:\.\d+)?\s*\)|[+-]?\d[\d,]*(?:\.\d+)?)', quote.replace('−', '-'))
            observed = [parse_number(x)[0] for x in mentions]
            if value not in observed:
                raise ValueError('Operand value missing from quote')
            normalize_unit(operand.get('unit', ''))
            normalize_scale(operand.get('scale', ''))
            if not operand.get('unit') or not operand.get('scale'):
                raise ValueError('Operand unit and scale required')
            values[name] = value
        def evaluate(node):
            if isinstance(node, ast.Expression): return evaluate(node.body)
            if isinstance(node, ast.Name): return values[node.id]
            if isinstance(node, ast.Constant) and type(node.value) in (int, float): return scalar(str(node.value))
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                return evaluate(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                left, right = evaluate(node.left), evaluate(node.right)
                if isinstance(node.op, ast.Add): return left + right
                if isinstance(node.op, ast.Sub): return left - right
                if isinstance(node.op, ast.Mult): return left * right
                return left / right
            raise ValueError('Only named operands, constants and + - * / are supported')
        result = evaluate(tree)
        scalar(format(result, 'f'))
        # Constants such as the 2 in an average are legitimate. At scoring time
        # the entire expression (including every constant) must match the
        # reviewed derivation, so invented constants cannot earn provenance.
        valid = bool(names) and any(isinstance(n, ast.BinOp) for n in ast.walk(tree))
        key = f'c{len(self.calculations) + 1}'
        record = {'calc_id': key, 'expression': expression, 'operands': copy.deepcopy(operands), 'value': format(result, 'f'), 'provenance_valid': valid, 'delivered_turn': self.turn}
        self.calculations[key] = record
        return record


def score_submission(target, submission, state, retrieval_weight=0.0):
    validate_target(target)
    result = dict(F=0.0, A=0.0, G=0.0, Ret=0.0, reward=0.0, correct=0, grounded_success=0,
                  unresolved=False, fabricated_citation=False, grader_version=SCORER_VERSION)
    if not submission: return result
    try: submission = validate_submission(submission)
    except (ValueError, KeyError): return result
    result['F'] = 1.0
    if submission['answer_type'] != target['answer_type']: return result
    kind = target['answer_type']
    if kind == 'numeric':
        result['A'] = float(numbers_equal(typed_number(target['value'], target['unit'], target['scale']), typed_number(submission['value'], submission['unit'], submission['scale']), target['precision']))
    elif kind == 'boolean':
        result['A'] = float(submission['decision'] == target['decision'])
    else:
        norm = lambda s: ' '.join(s.lower().split()).rstrip('.')
        exact = norm(submission['answer_text']) in {norm(x) for x in target.get('aliases', [])}
        result['A'] = float(exact)
        result['unresolved'] = not exact
    citations = submission.get('citations', [])
    visible = []
    for citation in citations:
        receipt = state.receipts.get(citation['receipt_id'])
        if not receipt or receipt.get('delivered_turn', -1) >= state.turn or receipt.get('document_id') != citation['document_id'] or receipt.get('page') != citation['page']:
            result['fabricated_citation'] = True
        elif receipt.get('tool') in ('read', 'read_table', 'grep_document'):
            visible.append(receipt)
    # Reviewed supporting spans distinguish relevant support from mere page access.
    supports = target.get('support', [])
    def supports_target(receipt):
        return any(s.get('quote') and s['document_id'] == receipt.get('document_id') and s['page'] == receipt.get('page') and s['quote'] in receipt.get('text', '') for s in supports)
    result['Ret'] = float(any(supports_target(r) for r in state.receipts.values()))
    claims = {s.get('claim', 'answer') for s in supports}
    grounding = bool(visible) and bool(claims) and all(any(s.get('claim', 'answer') == claim and s.get('quote') and s['document_id'] == r.get('document_id') and s['page'] == r.get('page') and s['quote'] in r.get('text', '') for s in supports for r in visible) for claim in claims)
    if kind == 'numeric' and target.get('derived'):
        calculation = state.calculations.get(submission.get('calc_id'), {})
        cited = {r['receipt_id'] for r in visible}
        grounding = calculation.get('delivered_turn', state.turn) < state.turn and bool(calculation.get('provenance_valid')) and all(o['receipt_id'] in cited for o in calculation.get('operands', {}).values())
        if grounding:
            grounding = numbers_equal(typed_number(calculation['value'], target['unit'], target['scale']), typed_number(submission['value'], submission['unit'], submission['scale']), target['precision'])
        # Reviewed derivation is necessary to establish metric semantics, not only numeric coincidence.
        expected_operands = target.get('operands', {})
        actual_operands = calculation.get('operands', {})
        def operand_signature(operand):
            value, unit, scale = typed_number(operand['value'], operand['unit'], operand['scale'])
            return (str(value.normalize()), unit, scale, str(operand.get('metric', '')).lower(), str(operand.get('period', '')))
        expected_signatures = {name: operand_signature(o) for name, o in expected_operands.items()}
        actual_signatures = {name: operand_signature(o) for name, o in actual_operands.items()}
        def expression_signature(expression, operands):
            class Substitute(ast.NodeTransformer):
                def visit_Name(self, node):
                    return ast.Constant(value=repr(operands.get(node.id)))
            return ast.dump(Substitute().visit(ast.parse(expression, mode='eval')))
        grounding = grounding and sorted(expected_signatures.values()) == sorted(actual_signatures.values())
        if grounding:
            for name, operand in actual_operands.items():
                matching = [o for key, o in expected_operands.items() if expected_signatures[key] == actual_signatures[name]]
                receipt = state.receipts.get(operand['receipt_id'], {})
                grounding = grounding and any(s.get('quote') and s['document_id'] == receipt.get('document_id') and s['page'] == receipt.get('page') and s['quote'] in receipt.get('text', '') and s['quote'] in operand.get('quote','') for o in matching for s in o.get('support', []))
        if grounding:
            grounding = expression_signature(target['expression'], expected_signatures) == expression_signature(calculation['expression'], actual_signatures)
    result['G'] = float(grounding and not result['fabricated_citation'])
    result['correct'] = int(result['A'] == 1 and not result['unresolved'])
    result['grounded_success'] = int(result['correct'] and result['G'] == 1)
    if not 0 <= retrieval_weight <= 0.1: raise ValueError('retrieval_weight must be between 0 and 0.1')
    if not result['fabricated_citation']:
        result['reward'] = (1-retrieval_weight) * result['A'] * (0.5 + 0.5*result['G']) + retrieval_weight*result['Ret']
    return result
