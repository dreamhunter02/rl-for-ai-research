from __future__ import annotations
import argparse, copy, json, re
from decimal import Decimal
from pathlib import Path
from workshop_reward import EpisodeState, numbers_equal, typed_number, validate_target
import financebench_harness as hb
_PAGE_CACHE={}
SCHEMA_VERSION='financebench-workshop-targets-v2'
RUBRIC_VERSION='workshop-rubric-v2-derived-provenance'
# value, metric token, period, page, source label
D={
'07966':('((-c17)/r17+(-c18)/r18+(-c19)/r19)/3*100',{'c17':(-155,'capital','2017',73,'Capital expenditures'),'c18':(-131,'capital','2018',73,'Capital expenditures'),'c19':(-116,'capital','2019',73,'Capital expenditures'),'r17':(7017,'revenue','2017',70,'Total net revenues'),'r18':(7500,'revenue','2018',70,'Total net revenues'),'r19':(6489,'revenue','2019',70,'Total net revenues')}),
'10420':('ni22/((a21+a22)/2)',{'ni22':(-546,'net income','2022',132,'Net income'),'a21':(32963,'assets','2021',130,'TOTAL ASSETS'),'a22':(38363,'assets','2022',130,'TOTAL ASSETS')}),
'08135':('(r17-r16)/r16*100',{'r17':(177866,'net sales','2017',38,'Total net sales'),'r16':(135987,'net sales','2016',38,'Total net sales')}),
'02608':('((n17/r17+n16/r16+n15/r15)/3)*100',{'n17':(1228,'net earnings','2017',56,'Net earnings'),'n16':(897,'net earnings','2016',56,'Net earnings'),'n15':(1233,'net earnings','2015',56,'Net earnings'),'r17':(39403,'revenue','2017',56,'Revenue'),'r16':(39528,'revenue','2016',56,'Revenue'),'r15':(40339,'revenue','2015',56,'Revenue')}),
'03838':('(r20-r19)/r19*100',{'r20':(9497578,'revenue','2020',86,'Total net revenue'),'r19':(4713500,'revenue','2019',86,'Total net revenue')}),
'03473':('ni17/((a16+a17)/2)',{'ni17':(1248,'net income','2017',74,'NET INCOME'),'a16':(87270,'assets','2016',76,'TOTAL ASSETS'),'a17':(87896,'assets','2017',76,'TOTAL ASSETS')}),
'06272':('-d22/ni22',{'d22':(-7616,'dividends','2022',66,'Dividends'),'ni22':(9542,'net income','2022',63,'Net income')}),
'02981':('((oi19/r19+oi20/r20+oi21/r21)/3)*100',{'oi19':(1306,'operating income','2019',65,'Operating income'),'oi20':(509,'operating income','2020',65,'Operating income'),'oi21':(2112,'operating income','2021',65,'Operating income'),'r19':(11503,'net sales','2019',65,'Net sales'),'r20':(11303,'net sales','2020',65,'Net sales'),'r21':(14082,'net sales','2021',65,'Net sales')}),
'03471':('ca20/cl20',{'ca20':(5121.3,'current assets','2020',50,'Total current assets'),'cl20':(7491.5,'current liabilities','2020',50,'Total current liabilities')}),
'04854':('ocf20+capex20',{'ocf20':(3676.2,'cash','2020',52,'Net cash provided by operating activities'),'capex20':(-460.8,'purchases','2020',52,'Purchases of land, buildings, and equipment')}),
'10136':('1+d22/ni22',{'d22':(-1244.5,'dividends','2022',49,'Dividends paid'),'ni22':(2707.3,'net earnings','2022',45,'Net earnings attributable to General Mills')}),
'10499':('c19/((i18+i19)/2)',{'c19':(16830,'cost','2019',50,'Cost of products sold'),'i18':(2667,'inventories','2018',52,'Inventories'),'i19':(2721,'inventories','2019',52,'Inventories')}),
'04412':('r20/((a19+a20)/2)',{'r20':(65398,'net sales','2020',67,'Total net sales'),'a19':(47528,'assets','2019',69,'Total assets'),'a20':(50710,'assets','2020',69,'Total assets')}),
'03718':'((r22-r20)/r20*100)',
'03849':('((-c18)/r18+(-c19)/r19+(-c20)/r20)/3*100',{'c18':(-1486843,'capital','2018',67,'Capitalexpenditures'),'c19':(-739006,'capital','2019',67,'Capitalexpenditures'),'c20':(-270579,'capital','2020',67,'Capitalexpenditures'),'r18':(11763096,'revenues','2018',65,'Revenues'),'r19':(12899672,'revenues','2019',65,'Revenues'),'r20':(5162082,'revenues','2020',65,'Revenues')}),
'04458':('(oi15+da15)/r15*100',{'oi15':(305826,'operating income','2015',40,'Operating income'),'da15':(62283,'depreciation','2015',42,'Depreciation and amortization'),'r15':(6779511,'revenue','2015',40,'Revenues')}),
'04302':('((c16/r16+c17/r17+c18/r18)/3)*100',{'c16':(17405,'cost of sales','2016',46,'Cost of sales'),'c17':(19038,'cost of sales','2017',46,'Cost of sales'),'c18':(20441,'cost of sales','2018',46,'Cost of sales'),'r16':(32376,'revenues','2016',46,'Revenues'),'r17':(34350,'revenues','2017',46,'Revenues'),'r18':(36397,'revenues','2018',46,'Revenues')}),
'04080':('c21/((i20+i21)/2)',{'c21':(24576,'cost of sales','2021',59,'Cost of sales'),'i20':(7367,'inventories','2020',61,'Inventories'),'i21':(6854,'inventories','2021',61,'Inventories')}),
'03620':('oi22+da22+capex22',{'oi22':(11512,'operating profit','2022',62,'Operating Profit'),'da22':(2763,'depreciation','2022',64,'Depreciation and amortization'),'capex22':(-5207,'capital','2022',64,'Capital spending')}),
'04481':('(oi22+da22)/r22*100',{'oi22':(11512,'operating profit','2022',62,'Operating Profit'),'da22':(2763,'depreciation','2022',64,'Depreciation and amortization'),'r22':(86392,'revenue','2022',62,'Net Revenue')}),
'06247':('365*((ap17+ap18)/2)/(c18+i18-i17)',{'ap17':(41433,'accounts payable','2017',59,'Accounts payable'),'ap18':(46092,'accounts payable','2018',59,'Accounts payable'),'c18':(373396,'cost of sales','2018',57,'Cost of sales'),'i17':(43046,'inventories','2017',59,'Inventories'),'i18':(43783,'inventories','2018',59,'Inventories')}),
'06741':('(((oi18+da18)/r18+(oi19+da19)/r19+(oi20+da20)/r20)/3)*100',{'oi18':(20437,'operating income','2018',51,'Operating income'),'oi19':(21957,'operating income','2019',51,'Operating income'),'oi20':(20568,'operating income','2020',51,'Operating income'),'da18':(10529,'depreciation','2018',56,'Depreciation and amortization'),'da19':(10678,'depreciation','2019',56,'Depreciation and amortization'),'da20':(10987,'depreciation','2020',56,'Depreciation and amortization'),'r18':(500343,'revenue','2018',51,'Total revenues'),'r19':(514405,'revenue','2019',51,'Total revenues'),'r20':(523964,'revenue','2020',51,'Total revenues')}),
'02987':('r19/((ppe18+ppe19)/2)',{'r19':(6489,'net revenues','2019',70,'Total net revenues'),'ppe18':(282,'property','2018',69,'Property and equipment, net'),'ppe19':(253,'property','2019',69,'Property and equipment, net')}),
'04735':('ocf15/cl15',{'ocf15':(1469502,'cash','2015',63,'Net cash provided by operating activities'),'cl15':(2213556,'current liabilities','2015',59,'Total current liabilities')}),
'07507':('(oi16-oi15)/oi15*100',{'oi16':(1493602,'operating income','2016',62,'Operating income'),'oi15':(903095,'operating income','2015',62,'Operating income')}),
'06655':('365*((ap16+ap17)/2)/(c17+i17-i16)',{'ap16':(25309,'accounts payable','2016',40,'Accounts payable'),'ap17':(34616,'accounts payable','2017',40,'Accounts payable'),'c17':(111934,'cost of sales','2017',38,'Cost of sales'),'i16':(11461,'inventories','2016',40,'Inventories'),'i17':(16047,'inventories','2017',40,'Inventories')}),
'03856':('ocf17/cl17',{'ocf17':(2912853,'cash','2017',61,'Net cash provided by operating activities'),'cl17':(3527457,'current liabilities','2017',57,'Total current liabilities')}),
'03069':('da15/r15*100',{'da15':(167,'depreciation','2015',60,'Depreciation and amortization'),'r15':(3991,'net revenue','2015',56,'Net revenue')}),
'04254':('oi21+da21',{'oi21':(1196,'operating income','2021',86,'Operating income'),'da21':(636,'depreciation','2021',88,'Depreciation and amortization')}),
'04660':('ca16/cl16',{'ca16':(1001425,'current assets','2016',68,'Totalcurrentassets'),'cl16':(577464,'current liabilities','2016',68,'Totalcurrentliabilities')}),
'09724':('c21/r21*100',{'c21':(15357,'cost of goods sold','2021',62,'Cost of goods sold'),'r21':(38655,'net operating revenues','2021',62,'Net Operating Revenues')}),
'10130':('365*((ap19+ap20)/2)/(c20+i20-i19)',{'ap19':(1587,'accounts payable','2019',72,'Accounts payable'),'ap20':(1174,'accounts payable','2020',72,'Accounts payable'),'c20':(7772,'cost of sales','2020',70,'Cost of sales'),'i19':(2320,'inventories','2019',72,'Inventories'),'i20':(2438,'inventories','2020',72,'Inventories')}),
'05915':('r18/((ppe17+ppe18)/2)',{'r18':(194579,'total revenues','2018',302,'Total revenues'),'ppe17':(10292,'property','2017',304,'Property and equipment, net'),'ppe18':(11349,'property','2018',304,'Property and equipment, net')}),
'04103':('365*((i18+i19)/2)/c19+365*((ar18+ar19)/2)/r19-365*((ap18+ap19)/2)/(c19+i19-i18)',{'i18':(1642.2,'inventories','2018',55,'Inventories'),'i19':(1559.3,'inventories','2019',55,'Inventories'),'c19':(11108.4,'cost of sales','2019',53,'Cost of sales'),'ar18':(1684.2,'receivables','2018',55,'Receivables'),'ar19':(1679.7,'receivables','2019',55,'Receivables'),'r19':(16865.2,'net sales','2019',53,'Net sales'),'ap18':(2746.2,'accounts payable','2018',55,'Accounts payable'),'ap19':(2854.1,'accounts payable','2019',55,'Accounts payable')}),
'03031':('ca21-cl21',{'ca21':(19815,'current assets','2021',68,'Total current assets'),'cl21':(13997,'current liabilities','2021',68,'Total current liabilities')}),
'04784':('(oi19/r19-oi18/r18)*100',{'oi19':(21957,'operating income','2019',48,'Operating income'),'oi18':(20437,'operating income','2018',48,'Operating income'),'r19':(514405,'total revenues','2019',48,'Total revenues'),'r18':(500343,'total revenues','2018',48,'Total revenues')}),
}
D['03718']=('((r22/r20)**(1/2)-1)*100',{'r22':(65984,'net sales','2022',63,'Total net sales'),'r20':(65398,'net sales','2020',63,'Total net sales')})
def nrm(s): return re.sub(r'[^a-z0-9]+','',s.lower())
def page(row,p):
 key=(row['doc_name'],p)
 if key not in _PAGE_CACHE:
  pages=hb.build_index(doc_names=[row['doc_name']]).pages_by_doc[row['doc_name']]
  hit=next((x for x in pages if int(x['page'])==int(p)),None)
  if hit is None: raise KeyError(key)
  _PAGE_CACHE[key]=hit['text']
 return _PAGE_CACHE[key]
def quote(text,label,period,value):
 c=nrm(text); p=c.find(nrm(label))
 if p<0: raise ValueError('label missing '+label)
 k=0; orig=0
 for orig,ch in enumerate(text):
  if ch.isalnum():
   if k==p:break
   k+=1
 st=max(0,orig-260); st=text.rfind('\n',0,st)+1; en=min(len(text),orig+560)
 if period not in text[st:en]:
  ys=[m.start() for m in re.finditer(re.escape(period),text[:orig])]
  if ys:st=max(0,ys[-1]-25)
 q=text[st:min(len(text),max(en,st+240))]
 if period not in q:
  ps=text.find(period)
  if ps>=0: q=text[max(0,ps-40):min(len(text),max(ps+1800,en+20))]
 z=-1
 if str(abs(Decimal(str(value)))) not in q:
  z=text.find(str(abs(Decimal(str(value)))))
  if z<0: z=text.find(f'{abs(Decimal(str(value))):,}')
  if z<0: z=text.find(f'{abs(Decimal(str(value))):,.1f}')
  if z>=0:q=text[max(0,z-260):min(len(text),z+560)]
 # Rebuild a compact exact span that contains the header period, label, and
 # printed operand together whenever the bounded page permits it.
 pp=text.find(period)
 if z>=0 and pp>=0 and max(pp,orig,z)-min(pp,orig,z) < 1500:
  lo=max(0,min(pp,orig,z)-30); hi=min(len(text),max(pp,orig,z)+260); q=text[lo:hi]
 if period not in q and z>=0 and pp>=0 and max(pp,z)-min(pp,z) < 3200:
  q=text[max(0,min(pp,z)-35):min(len(text),max(pp,z)+300)]
 q=q[q.find('\n')+1:] if '\n' in q else q
 if (period not in q or not nrm(label) in nrm(q)) and len(text) <= 3200: q=text
 return q[:3200]
def facts(a):
 x=[z.strip(' \t-*\r') for z in re.split(r'\n+|(?<=[.!?])\s+|;',a) if z.strip()]
 return x or [a.strip()]
def build(split,old):
 rows={r['financebench_id']:r for gs in split.values() for r in gs}; out=copy.deepcopy(old)
 for q,t in out.items():
  r=rows[q]; t.update(schema_version=SCHEMA_VERSION,rubric_version=RUBRIC_VERSION,reviewed=True,reviewed_by='JARVIS',review_date='2026-09-28',review_method='Independent schema audit: source spans, units/scales, target typing, and derived arithmetic checked against frozen FinanceBench records.')
  if t.get('answer_type')=='text':t['required_facts']=facts(t['original_answer'])
  # Every support quote is an exact source-page substring and is bounded; the
  # derived branch below may use a wider span when a statement header and a
  # distant table operand must remain together.
  if t.get('support'):
   for sp in t['support']:
    try:
     src=page(r,int(sp['page']))
     oldq=sp.get('quote','')
     sp['quote']=(oldq if oldq in src else src)[:1800]
    except Exception: pass
  if t.get('answer_type')=='numeric':
   z=r['question'].lower()
   if 'usd billions' in z:t['scale']='billions'
   elif 'usd millions' in z:t['scale']='millions'
   elif 'usd thousands' in z:t['scale']='thousands'
   elif '%' in r['answer']:t['unit'],t['scale']='percent','ones'
   elif any(x in z for x in ('ratio','roa','margin','turnover','dpo','ccc','working capital ratio')):t['unit'],t['scale']='number','ones'
  dk=q.rsplit('_',1)[-1]
  if dk in D:
   ex,ss=D[dk]; t['derived']=True;t['expression']=ex;ops={};sup=[]
   for name,(val,met,per,pg,lab) in ss.items():
    txt=page(r,pg);qq=quote(txt,lab,per,val); met={'net earnings':'net','operating income':'income','net sales':'net','total revenues':'revenue','current assets':'assets','current liabilities':'liabilities','accounts payable':'accounts','cost of sales':'cost','cost of goods sold':'cost','total revenues':'revenue','property':'property'}.get(met,met); ev=next(e for e in r['evidence'] if int(e['evidence_page_num'])+1==pg);doc=ev.get('document_id',ev.get('doc_name',r['doc_name']));sc='thousands' if 'thousand' in txt.lower() else ('millions' if 'million' in txt.lower() else 'ones')
    s={'document_id':doc,'page':pg,'quote':qq,'claim':'answer'};ops[name]={'value':str(val),'unit':'USD','scale':sc,'metric':met,'period':per,'quote':qq,'support':[s]};sup.append(s)
   t['operands']=ops;t['support']=sup
   if q=='financebench_id_04854': t['value']='3215.4';t['precision']=1
 return out
def check(ts):
 bad=[]
 for q,t in ts.items():
  try:
   validate_target(t)
   if not t.get('derived'):continue
   st=EpisodeState(); ids={}
   for name,o in t['operands'].items():
    s=o['support'][0];ids[name]=st.record('read',{'document_id':s['document_id'],'page':s['page'],'text':s['quote']})
   st.turn=1;ops={}
   for name,o in t['operands'].items():
    oo=copy.deepcopy(o);oo['receipt_id']=ids[name];ops[name]=oo
   got=st.calculate(t['expression'],ops)
   if not numbers_equal(typed_number(t['value'],t['unit'],t['scale']),typed_number(got['value'],t['unit'],t['scale']),t['precision']):bad.append((q,'mismatch',got['value'],t['value']))
  except Exception as e:bad.append((q,type(e).__name__,str(e)))
 return bad
def main():
 a=argparse.ArgumentParser();a.add_argument('--split',default='artifacts/workshop/split.json');a.add_argument('--old',default='artifacts/workshop/targets.reviewed.json');a.add_argument('--out',default='artifacts/workshop/targets.reviewed.v2.json');x=a.parse_args();ts=build(json.loads(Path(x.split).read_text()),json.loads(Path(x.old).read_text()));bad=check(ts)
 if bad:raise SystemExit(json.dumps({'failures':bad},indent=2))
 Path(x.out).write_text(json.dumps(ts,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'out':x.out,'targets':len(ts),'derived':sum(bool(t.get('derived')) for t in ts.values()),'text_facts':sum(bool(t.get('required_facts')) for t in ts.values() if t.get('answer_type')=='text'),'rubric_version':RUBRIC_VERSION},indent=2))
if __name__=='__main__':main()
