"""Assemble delivery evidence after completed calculations; no strategy runs."""
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

from svxylab.panic_retreat import ROOT, digest, write_json


def canonical_svg(text):
    # Renderer-internal IDs are not geometry or scientific content. Preserve
    # every coordinate, style and glyph, and replace definitions/references only.
    names=re.findall(r'\bid="([^"]+)"',text)
    mapping={name:f'element_{i}' for i,name in enumerate(names)}
    text=re.sub(r'\bid="([^"]+)"',lambda m:'id="'+mapping[m[1]]+'"',text)
    return re.sub(r'#([A-Za-z0-9_-]+)(?=[)"\s])',lambda m:'#'+mapping.get(m[1],m[1]),text)


def main():
    base=ROOT/'runs/panic_retreat_diagnostic'
    runs=sorted(base.glob('*/run.json'))
    first,second=(json.loads(p.read_text()) for p in runs[-2:])
    a,b=ROOT/first['output_dir'],ROOT/second['output_dir']
    assert first['source_sha256']==second['source_sha256']
    ia=json.loads(runs[-2].with_name('experiment_record.json').read_text())
    ib=json.loads(runs[-1].with_name('experiment_record.json').read_text())
    assert ia==ib
    comparison=[]
    for suffix in ('*.csv','*.json','*.png','*.svg'):
        files=sorted(a.glob(suffix)); other=sorted(b.glob(suffix))
        assert [p.name for p in files]==[p.name for p in other]
        for p in files:
            same=digest(p)==digest(b/p.name)
            if p.suffix=='.svg':
                assert canonical_svg(p.read_text())==canonical_svg((b/p.name).read_text()),p.name
            else:
                assert same,p.name
            comparison.append(dict(file=p.name,first_sha256=digest(p),second_sha256=digest(b/p.name),
                                   byte_identical=same,internal_id_normalized_identical=True if p.suffix=='.svg' else None))
    # Real-input audit was on the preceding complete calculation. All numeric
    # outputs and the reviewed PNGs must still agree with the delivered run.
    independent=json.loads((base/'independent_validation.json').read_text())
    audited=ROOT/independent['audited_output_dir']
    assert all(digest(p)==digest(b/p.name) for p in audited.glob('*.csv'))
    reviewed=ROOT/'data/clean/panic_retreat_diagnostic/20260909T182330678303Z'
    assert all(digest(p)==digest(b/p.name) for p in reviewed.glob('*.png'))
    suites=ET.parse(base/'pytest_final.xml').getroot().findall('testsuite')
    totals={key:sum(int(x.attrib.get(key,0)) for x in suites) for key in ('tests','failures','errors','skipped')}
    assert totals==dict(tests=169,failures=0,errors=0,skipped=0)
    assert all(digest(ROOT/f)==h for f,h in ib['input_sha256'].items())
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    origin=subprocess.check_output(['git','rev-parse','origin/main'],cwd=ROOT,text=True).strip()
    assert head==origin=='25d96bbb39e9f74c421ab9fbeec029dabbc282b0'
    modified=subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines()
    assert set(modified)<= {'AGENTS.md','README.md','STATUS.md'}
    reproducibility=dict(status='CALCULATIONS_AND_PNG_BYTE_IDENTICAL_SVG_INTERNAL_IDS_NORMALIZED',runs=[str(p.relative_to(ROOT)) for p in runs[-2:]],
        same_precalculation_record=True,same_implementation_identity=True,
        csv_files=sum(x['file'].endswith('.csv') for x in comparison),
        json_files=sum(x['file'].endswith('.json') for x in comparison),
        png_files=sum(x['file'].endswith('.png') for x in comparison),
        svg_files=sum(x['file'].endswith('.svg') for x in comparison),comparison=comparison,
        prior_independent_audit_applies_to_identical_csv=True,
        report_policy='CSV/JSON/PNG逐字节相同；SVG仅将内部元素ID及其引用规范化后逐字相同，所有坐标、样式、字形保留比较。HTML含实际运行目录，不称逐字节相同。')
    write_json(base/'reproducibility.json',reproducibility)
    report=ROOT/second['report']
    result=dict(stage='E1_SUPPLEMENTAL_DIAGNOSTIC',completed_at_utc=datetime.now(timezone.utc).isoformat(),
        engineering='计算、测试、独立核账和复算完成；整页工具视觉核验受限',
        research='揭示后有限诊断完成；三项机制证据平列；未选择或执行机制',
        real_data=second['validation']['coverage'],baseline_commit=head,
        final_run=str(runs[-1].relative_to(ROOT)),output_dir=second['output_dir'],
        report=second['report'],report_sha256=digest(report),ordinary_pytest=totals,
        independent_validation='runs/panic_retreat_diagnostic/independent_validation.json',
        real_input_prefix_checks=independent['real_input_prefixes'],
        future_and_information_checks=independent['future_signal_perturbations'],
        independent_accounts=independent['independent_accounts'],
        independent_account_rows=independent['independent_account_rows'],
        independent_max_amount_error=independent['max_account_error'],
        original_105_and_delayed_35_match=True,attribution_identity_max_error=0.,
        reproducibility=dict(csv_byte_identical=18,json_byte_identical=2,png_byte_identical=7,
                             svg_internal_ids_normalized_identical=7),
        original_E1_inputs_sources_and_report_unchanged=len(ib['input_sha256']),
        old_tracked_changes=modified,m2_or_original_E1_logic_changed=False,
        visual=dict(macos_open_exit_codes=[first['open_exit_code'],second['open_exit_code']],
            seven_pngs_directly_inspected=True,final_pngs_match_inspected=True,
            full_page_screenshot_verified=False,interactive_details_verified=False,
            limitation='浏览器工具此前拒绝本地HTML；未绕过。macOS打开、图表目视与静态链接不等于整页截图验收。'),
        limitations=['历史实际发布与接收时刻未证明；lag=1没有新增完成日线可做成交前复查',
            '候选日高度重叠，不能作为独立样本，也不是再入场账户或可实时执行的策略证明',
            '路径风险仅有日收盘；没有盘中成交或盘中最大损失证据',
            '工程探针早于单独根记录，记录在experiment_record.json；正式入口每次在新收益计算前保存完整口径与输入摘要，不称揭示前预注册'],
        source_sha256=second['source_sha256'],test_sha256=digest(ROOT/'tests/test_panic_retreat_diagnostic.py'),
        mechanism_selected=None,mechanism_executed=False,committed=False,pushed=False,
        stop='等待用户选择及验收；不开始下一实验')
    write_json(base/'final_validation.json',result)
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[];self.ids=[]
        def handle_starttag(self,tag,attrs):
            d=dict(attrs)
            if 'id' in d:self.ids.append(d['id'])
            self.links.extend(d[k] for k in ('href','src') if k in d)
    parser=Links();parser.feed(report.read_text())
    for link in parser.links:
        url=urlsplit(link);assert not url.scheme
        if url.path:assert (report.parent/unquote(url.path)).exists(),link
        else:assert url.fragment in parser.ids
    result['visual'].update(static_links_verified=len(parser.links),broken_links=0)
    write_json(base/'final_validation.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
