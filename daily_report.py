"""
株式定点観察レポート - 毎日自動生成スクリプト
対象銘柄はconfig.jsonで管理
"""

import os
import json
import time
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from anthropic import Anthropic

# ─── パス設定 ─────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
CONFIG_FILE = BASE_DIR / "config.json"
PREV_FILE   = BASE_DIR / "data" / "previous.json"
REPORT_DIR  = BASE_DIR / "reports"
NOTE_DIR    = BASE_DIR / "note"
INDEX_FILE  = BASE_DIR / "index.html"

REPORT_DIR.mkdir(exist_ok=True)
NOTE_DIR.mkdir(exist_ok=True)
(BASE_DIR / "data").mkdir(exist_ok=True)

# ─── 設定読み込み ─────────────────────────────────────────────────
def load_config():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# ─── 前日データ ───────────────────────────────────────────────────
def load_previous():
    if PREV_FILE.exists():
        with open(PREV_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_previous(data):
    with open(PREV_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ─── テクニカル指標 ───────────────────────────────────────────────
def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return float((100 - (100 / (1 + rs))).iloc[-1])

def calc_macd(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    hist  = macd - sig
    return float(macd.iloc[-1]), float(sig.iloc[-1]), float(hist.iloc[-1]), float(hist.iloc[-2])

def calc_bollinger(close, window=20):
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    return float((mid + 2*std).iloc[-1]), float(mid.iloc[-1]), float((mid - 2*std).iloc[-1])

def fetch_indicators(ticker):
    for attempt in range(3):
        try:
            df = yf.download(ticker, period="6mo", progress=False, auto_adjust=True)
            if not df.empty:
                info = yf.Ticker(ticker).info
                name = info.get("longName") or info.get("shortName") or ticker
                close  = df["Close"].squeeze()
                volume = df["Volume"].squeeze()
                price      = float(close.iloc[-1])
                prev_price = float(close.iloc[-2])
                bb_upper, bb_mid, bb_lower = calc_bollinger(close)
                ind = {
                    "ticker":     ticker,
                    "name":       name,
                    "price":      price,
                    "prev_price": prev_price,
                    "change_pct": (price - prev_price) / prev_price * 100,
                    "ma5":        float(close.rolling(5).mean().iloc[-1]),
                    "ma25":       float(close.rolling(25).mean().iloc[-1]),
                    "ma75":       float(close.rolling(75).mean().iloc[-1]) if len(close) >= 75 else float("nan"),
                    "rsi":        calc_rsi(close),
                    "macd":       calc_macd(close)[0],
                    "macd_sig":   calc_macd(close)[1],
                    "macd_hist":  calc_macd(close)[2],
                    "macd_prev":  calc_macd(close)[3],
                    "bb_upper":   bb_upper,
                    "bb_mid":     bb_mid,
                    "bb_lower":   bb_lower,
                    "volume":     float(volume.iloc[-1]),
                    "volume_ma5": float(volume.rolling(5).mean().iloc[-1]),
                }
                return ind
        except Exception as e:
            print(f"  [{ticker}] attempt {attempt+1} failed: {e}")
        time.sleep(2)
    return None

# ─── Claude 観察レポート生成 ──────────────────────────────────────
def generate_report(client, ind, prev_ind):
    if prev_ind:
        diff_text = f"""
【前日からの変化】
・株価: {prev_ind['price']:.2f} → {ind['price']:.2f}（{ind['change_pct']:+.2f}%）
・RSI: {prev_ind['rsi']:.1f} → {ind['rsi']:.1f}（{ind['rsi']-prev_ind['rsi']:+.1f}）
・MACDヒスト: {prev_ind['macd_hist']:.3f} → {ind['macd_hist']:.3f}
・出来高: {prev_ind['volume']:.0f} → {ind['volume']:.0f}
"""
    else:
        diff_text = "【前日データ】初回観察のため比較なし"

    prompt = f"""あなたは株式マーケットの観察記録を担当するアナリストです。
投資推奨ではなく、銘柄の状態を客観的に記録・観察するレポートを日本語で書いてください。

【銘柄】{ind['ticker']}（{ind['name']}）
【現在値】{ind['price']:.2f}（前日比 {ind['change_pct']:+.2f}%）
【移動平均線】MA5={ind['ma5']:.2f} / MA25={ind['ma25']:.2f} / MA75={ind['ma75']:.2f}
【RSI(14)】{ind['rsi']:.1f}
【MACD】ライン={ind['macd']:.3f} / シグナル={ind['macd_sig']:.3f} / ヒスト={ind['macd_hist']:.3f}（前日={ind['macd_prev']:.3f}）
【ボリンジャーバンド】上限={ind['bb_upper']:.2f} / 中央={ind['bb_mid']:.2f} / 下限={ind['bb_lower']:.2f}
【出来高】直近={ind['volume']:.0f} / 5日平均={ind['volume_ma5']:.0f}
{diff_text}

以下の構成で観察レポートを書いてください。
「エントリー」「利確」「損切り」などの売買用語は使わず、観察・記録の視点で書いてください：

1. **総合観察**：現在の状態を「🟢 上昇傾向」「🔴 下落傾向」「⚪ 横ばい」で示し、その強度（強／中／弱）を添える
2. **各指標の観察**：各指標が示している状況を箇条書きで記録
3. **前日からの変化**：昨日と比べて何が変わったか、どんな動きが見られるか
4. **今日の注目ポイント**：特に気になる指標や水準、継続して観察すべき点
5. **リスク観察**：警戒しておくべき兆候や状況

※ このレポートは観察記録です。"""

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text

# ─── note投稿用テキスト生成 ───────────────────────────────────────
def build_note_text(date_str, results):
    today = datetime.date.today()
    names = "・".join([r["ind"]["name"] for r in results])

    lines = []

    # タイトル候補（noteに貼るときの参考）
    lines.append(f"【noteタイトル候補】")
    lines.append(f"📊 株式定点観察レポート｜{date_str}")
    lines.append("")
    lines.append("=" * 50)
    lines.append("【ここから本文をコピーしてnoteに貼ってください】")
    lines.append("=" * 50)
    lines.append("")

    # 本文
    lines.append(f"📊 株式定点観察レポート　{date_str}")
    lines.append("")
    lines.append(f"本日の観察銘柄：{names}")
    lines.append("")

    for r in results:
        ind    = r["ind"]
        report = r["report"]
        change_sign = "▲" if ind["change_pct"] >= 0 else "▼"
        change_abs  = abs(ind["change_pct"])

        lines.append("━" * 30)
        lines.append(f"■ {ind['ticker']}　{ind['name']}")
        lines.append("")
        lines.append(f"現在値：{ind['price']:,.2f}円　{change_sign}{change_abs:.2f}%")
        lines.append(f"RSI：{ind['rsi']:.1f}　／　MA5：{ind['ma5']:,.2f}　／　MA25：{ind['ma25']:,.2f}")
        lines.append(f"MACDヒスト：{ind['macd_hist']:.3f}")
        lines.append("")
        lines.append(report)
        lines.append("")

    lines.append("━" * 30)
    lines.append("")
    lines.append("※ このレポートはAIによるテクニカル指標の観察記録です。")
    lines.append("　投資判断の根拠にはしないでください。")
    lines.append("")
    lines.append(f"#株式観察 #テクニカル分析 #定点観測 #{today.strftime('%Y%m%d')}")

    return "\n".join(lines)

# ─── HTML生成 ─────────────────────────────────────────────────────
def build_report_html(date_str, results):
    cards = ""
    for r in results:
        ind    = r["ind"]
        report = r["report"]
        change_class = "up" if ind["change_pct"] >= 0 else "down"
        change_sign  = "+" if ind["change_pct"] >= 0 else ""
        report_html  = report.replace("\n", "<br>")

        cards += f"""
        <div class="card">
          <div class="card-header">
            <div class="ticker-info">
              <span class="ticker">{ind['ticker']}</span>
              <span class="name">{ind['name']}</span>
            </div>
            <div class="price-info">
              <span class="price">{ind['price']:,.2f}</span>
              <span class="change {change_class}">{change_sign}{ind['change_pct']:.2f}%</span>
            </div>
          </div>
          <div class="metrics">
            <div class="metric"><span class="label">RSI</span><span class="value">{ind['rsi']:.1f}</span></div>
            <div class="metric"><span class="label">MA5</span><span class="value">{ind['ma5']:,.2f}</span></div>
            <div class="metric"><span class="label">MA25</span><span class="value">{ind['ma25']:,.2f}</span></div>
            <div class="metric"><span class="label">MACDヒスト</span><span class="value">{ind['macd_hist']:.3f}</span></div>
          </div>
          <div class="report-body">{report_html}</div>
        </div>
"""

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>観察レポート {date_str}</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap');
    :root {{
      --bg: #0d1117; --surface: #161b22; --border: #30363d;
      --text: #e6edf3; --muted: #8b949e; --up: #3fb950;
      --down: #f85149; --accent: #58a6ff;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: var(--bg); color: var(--text); font-family: 'Noto Sans JP', sans-serif; padding: 2rem 1rem; }}
    .container {{ max-width: 860px; margin: 0 auto; }}
    header {{ border-bottom: 1px solid var(--border); padding-bottom: 1.5rem; margin-bottom: 2rem; }}
    header h1 {{ font-size: 1.1rem; color: var(--muted); font-weight: 400; letter-spacing: 0.05em; }}
    header h2 {{ font-size: 1.8rem; font-weight: 700; margin-top: 0.3rem; }}
    .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; margin-bottom: 1.5rem; overflow: hidden; }}
    .card-header {{ display: flex; justify-content: space-between; align-items: center; padding: 1.2rem 1.5rem; border-bottom: 1px solid var(--border); }}
    .ticker {{ font-family: 'JetBrains Mono', monospace; font-size: 1rem; font-weight: 600; color: var(--accent); }}
    .name {{ margin-left: 0.8rem; color: var(--muted); font-size: 0.9rem; }}
    .price {{ font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 600; }}
    .change {{ font-family: 'JetBrains Mono', monospace; font-size: 0.95rem; margin-left: 0.6rem; font-weight: 600; }}
    .change.up {{ color: var(--up); }} .change.down {{ color: var(--down); }}
    .metrics {{ display: flex; gap: 0; border-bottom: 1px solid var(--border); }}
    .metric {{ flex: 1; padding: 0.8rem 1rem; border-right: 1px solid var(--border); text-align: center; }}
    .metric:last-child {{ border-right: none; }}
    .metric .label {{ display: block; font-size: 0.7rem; color: var(--muted); letter-spacing: 0.05em; margin-bottom: 0.3rem; }}
    .metric .value {{ font-family: 'JetBrains Mono', monospace; font-size: 0.95rem; font-weight: 600; }}
    .report-body {{ padding: 1.5rem; line-height: 1.9; font-size: 0.92rem; color: #cdd9e5; white-space: pre-wrap; }}
    footer {{ text-align: center; color: var(--muted); font-size: 0.8rem; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--border); }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>📊 株式定点観察レポート</h1>
      <h2>{date_str}</h2>
    </header>
    {cards}
    <footer>
      <p>このレポートは観察記録です。投資判断の根拠にしないでください。</p>
      <p style="margin-top:0.5rem"><a href="../index.html">← レポート一覧へ</a></p>
    </footer>
  </div>
</body>
</html>"""
    return html

def build_index_html(report_files):
    items = ""
    for fname, date_str in report_files:
        items += f'<li><a href="reports/{fname}">📅 {date_str}</a></li>\n'

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>株式定点観察レポート 一覧</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&family=JetBrains+Mono&display=swap');
    :root {{ --bg:#0d1117; --surface:#161b22; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; }}
    * {{ box-sizing:border-box; margin:0; padding:0; }}
    body {{ background:var(--bg); color:var(--text); font-family:'Noto Sans JP',sans-serif; padding:3rem 1rem; }}
    .container {{ max-width:600px; margin:0 auto; }}
    h1 {{ font-size:1.6rem; margin-bottom:0.5rem; }}
    p {{ color:var(--muted); margin-bottom:2rem; font-size:0.9rem; }}
    ul {{ list-style:none; }}
    li {{ border-bottom:1px solid var(--border); }}
    li a {{ display:block; padding:1rem 0.5rem; color:var(--accent); text-decoration:none; font-size:1rem; transition:background 0.15s; }}
    li a:hover {{ background:var(--surface); border-radius:6px; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>📊 株式定点観察レポート</h1>
    <p>毎日自動更新 ・ 観察記録アーカイブ</p>
    <ul>
{items}
    </ul>
  </div>
</body>
</html>"""

# ─── メイン ───────────────────────────────────────────────────────
def main():
    config  = load_config()
    tickers = config["tickers"]
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY が設定されていません")

    client   = Anthropic(api_key=api_key)
    previous = load_previous()
    today    = datetime.date.today()
    date_str = today.strftime("%Y年%m月%d日")
    fname    = f"{today.strftime('%Y-%m-%d')}.html"

    results  = []
    new_prev = {}

    for ticker in tickers:
        print(f"[{ticker}] データ取得中...")
        ind = fetch_indicators(ticker)
        if ind is None:
            print(f"[{ticker}] 取得失敗。スキップします。")
            continue

        prev_ind = previous.get(ticker)
        print(f"[{ticker}] Claudeでレポート生成中...")
        report = generate_report(client, ind, prev_ind)

        results.append({"ind": ind, "report": report})
        new_prev[ticker] = ind
        time.sleep(1)

    # 今日のレポートHTML（GitHub Pages用）
    report_html = build_report_html(date_str, results)
    report_path = REPORT_DIR / fname
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_html)
    print(f"レポートHTML保存: {report_path}")

    # note投稿用テキスト
    note_text = build_note_text(date_str, results)
    note_path = NOTE_DIR / f"{today.strftime('%Y-%m-%d')}.txt"
    with open(note_path, "w", encoding="utf-8") as f:
        f.write(note_text)
    print(f"note用テキスト保存: {note_path}")

    # index.html更新
    existing = sorted(
        [(p.name, p.stem.replace("-", "年", 1).replace("-", "月", 1) + "日")
         for p in REPORT_DIR.glob("*.html")],
        reverse=True
    )
    index_html = build_index_html(existing)
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        f.write(index_html)

    # 前日データ更新
    save_previous(new_prev)
    print("完了!")

if __name__ == "__main__":
    main()
