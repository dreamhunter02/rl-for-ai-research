"""Mechanical source/display audit; this does not certify accounting semantics."""
from collections import Counter
import hashlib
from pathlib import Path

import financebench_harness as hb


def inspect_document(doc_name: str) -> dict:
    pdf = hb.FILINGS / f'{doc_name}.pdf'
    source = pdf if pdf.exists() else hb.TEXTDIR / f'{doc_name}.txt'
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    rows = hb.load_pages(doc_name)
    result = {
        'document_id': doc_name, 'source_kind': 'pdf' if pdf.exists() else 'text',
        'source_path': str(source.resolve()), 'source_sha256': source_hash,
        'pages': len(rows),
        'extraction_counts': dict(Counter(r.get('extraction_applied', 'plain') for r in rows)),
    }
    if [r['page'] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f'{doc_name}: discontinuous page identities')
    if pdf.exists():
        from pypdf import PdfReader
        result['pdf_pages'] = len(PdfReader(pdf).pages)
        if result['pdf_pages'] != len(rows):
            raise ValueError(f'{doc_name}: extraction changed the PDF page count')
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise ValueError(f'{doc_name}: source changed during extraction')
    return result
