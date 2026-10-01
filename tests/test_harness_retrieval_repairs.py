import financebench_harness as hb


def test_plural_income_title():
    assert hb._statement_kind('Company\nCONSOLIDATED STATEMENTS OF INCOME\n2018 2017')=='operations'


def test_notes_balance_reference_is_not_statement():
    assert hb._statement_kind('Notes to consolidated financial statements\nDerivatives are included in the balance sheets.')=='other'


def test_alias_without_fuzzy_company_leakage():
    i=hb.StructuredIndex([],[],[])
    assert i._allowed({'company':'AES'},{'company':'AES Corporation'})
    assert not i._allowed({'company':'AES Energy'},{'company':'AES Corporation'})


def test_grep_offsets_reconstruct_exact_page_slice():
    text='head\nrevenue 100\nfoot\n'
    i=hb.StructuredIndex([{'document_id':'D','page':1,'text':text}],[],[])
    h=i.grep_document('D',['revenue'],context_lines=0)[0]
    assert text[h['start']:h['end']]==h['text']
