"""
1002쌍 매칭 검증용 HTML 페이지 생성.
브라우저에서 좌(test) ↔ 우(ours) 비교, lazy loading, ID 검색 가능.
"""
import csv
import html
from pathlib import Path

BASE = Path('/home/student/GEN_AI_GTG')
OUT = BASE / 'detection/phash_matching'
CSV_PATH = OUT / 'fake_gt_matches.csv'
HTML_PATH = OUT / 'verification.html'

rows = list(csv.DictReader(open(CSV_PATH)))
# our_image의 ID 번호 기준으로 정렬 (ID_0_GT_frame1 → ID_0, frame1)
def sort_key(r):
    name = Path(r['our_image_path']).name  # ID_139_GT_frame1.png
    parts = name.replace('.png', '').split('_')
    try:
        return (int(parts[1]), parts[3])  # (139, 'frame1')
    except Exception:
        return (10**9, name)

rows.sort(key=sort_key)
N = len(rows)

# Live Preview/Simple Browser는 워크스페이스 루트에서 HTTP 서빙하므로
# verification.html에서 이미지로의 상대 경로를 만들어야 함.
HTML_DIR = OUT  # detection/phash_matching/
import os
def to_url(p):
    return os.path.relpath(p, HTML_DIR).replace(os.sep, '/')

cards = []
for i, r in enumerate(rows):
    test_p = r['test_image_path']
    our_p  = r['our_image_path']
    d      = r['hamming_distance']
    stage  = r['stage']
    test_name = Path(test_p).name
    our_name  = Path(our_p).name
    cards.append(f"""
    <div class="card" data-our="{html.escape(our_name)}" data-test="{html.escape(test_name)}">
      <div class="idx">{i+1}/{N}</div>
      <div class="pair">
        <div class="col">
          <img loading="lazy" src="{to_url(test_p)}" alt="{html.escape(test_name)}">
          <div class="cap test">TEST: {html.escape(test_name)}</div>
        </div>
        <div class="col">
          <img loading="lazy" src="{to_url(our_p)}" alt="{html.escape(our_name)}">
          <div class="cap ours">OURS: {html.escape(our_name)}</div>
        </div>
      </div>
      <div class="meta">d={d} · stage={stage}</div>
    </div>
    """)

cards_html = '\n'.join(cards)

doc = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>pHash 매칭 검증 — {N}쌍</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 16px;
    font-family: -apple-system, "Segoe UI", "Noto Sans KR", sans-serif;
    background: #1a1a1a; color: #e5e5e5;
  }}
  header {{
    position: sticky; top: 0; z-index: 10;
    background: #1a1a1a; padding: 10px 0 14px;
    border-bottom: 1px solid #333;
    display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
  }}
  h1 {{ margin: 0; font-size: 18px; font-weight: 600; }}
  .summary {{ font-size: 13px; color: #aaa; }}
  input[type=text] {{
    background: #2a2a2a; color: #fff; border: 1px solid #444;
    padding: 6px 10px; border-radius: 6px; font-size: 13px;
    min-width: 240px;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(540px, 1fr));
    gap: 14px; margin-top: 16px;
  }}
  .card {{
    background: #242424; border: 1px solid #333; border-radius: 8px;
    padding: 8px; position: relative;
  }}
  .idx {{
    position: absolute; top: 6px; right: 10px;
    font-size: 11px; color: #888;
  }}
  .pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }}
  .col {{ display: flex; flex-direction: column; }}
  .col img {{
    width: 100%; height: auto; aspect-ratio: 1;
    object-fit: cover; border-radius: 4px;
    background: #111;
  }}
  .cap {{
    font-size: 11px; padding: 4px 2px; word-break: break-all;
    font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
  }}
  .cap.test {{ color: #ffb86c; }}
  .cap.ours {{ color: #8be9fd; }}
  .meta {{
    font-size: 11px; color: #777; text-align: center; margin-top: 4px;
    font-family: ui-monospace, monospace;
  }}
  .hidden {{ display: none !important; }}
</style>
</head>
<body>
<header>
  <h1>pHash 매칭 검증</h1>
  <span class="summary">총 {N}쌍 · 좌=TEST(주최측 배포) · 우=OURS(우리 제출)</span>
  <input id="search" type="text" placeholder="ID_139 또는 test 해시로 필터…">
  <span id="counter" class="summary"></span>
</header>
<div class="grid" id="grid">
{cards_html}
</div>
<script>
  const N = {N};
  const search = document.getElementById('search');
  const counter = document.getElementById('counter');
  const cards = Array.from(document.querySelectorAll('.card'));
  function updateCounter(visible) {{
    counter.textContent = visible === N ? '' : `${{visible}} / ${{N}} 표시`;
  }}
  search.addEventListener('input', () => {{
    const q = search.value.trim().toLowerCase();
    let visible = 0;
    cards.forEach(c => {{
      if (!q || c.dataset.our.toLowerCase().includes(q) || c.dataset.test.toLowerCase().includes(q)) {{
        c.classList.remove('hidden'); visible++;
      }} else {{
        c.classList.add('hidden');
      }}
    }});
    updateCounter(visible);
  }});
  updateCounter(N);
</script>
</body>
</html>
"""
HTML_PATH.write_text(doc, encoding='utf-8')
print(f'저장: {HTML_PATH}')
print(f'  카드 수: {N}')
print(f'  파일 크기: {HTML_PATH.stat().st_size / 1024:.1f} KB')
print(f'\nVS Code에서 열려면:')
print(f'  code {HTML_PATH}')
print(f'또는 브라우저에서 직접:')
print(f'  file://{HTML_PATH}')
