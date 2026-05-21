"""
株式定点観察レポート - 毎日自動生成スクリプト
対象銘柄はconfig.jsonで管理
土日・日本市場の祝日は自動スキップ
"""

import os
import sys
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

# ─── 市場開場チェック ─────────────────────────────────────────────
def get_jp_holidays(year):
    holidays = set()
    fixed = [
        (1, 1), (2, 11), (2, 23), (4, 29),
        (5, 3), (5, 4), (5, 5), (8, 11),
        (11, 3), (11, 23), (12, 31),
    ]
    for m, d in fixed:
        holidays.add(datetime.date(year, m, d))

    def nth_monday(year, month, n):
        first = datetime.date(year, month, 1)
        first_monday = first + datetime.timedelta(days=(7 - first.weekday()) % 7)
        return first_monday + datetime.timedelta(weeks=n - 1)

    holidays.add(nth_monday(year, 1, 2))
    holidays.add(nth_monday(year, 7, 3))
    holidays.add(nth_monday(year, 9, 3))
    holidays.add(nth_monday(year, 10, 2))

    spring_day = 20 if year % 4 == 0 else 21
    autumn_day = 23 if year % 4 < 2 else 22
    holidays.add(datetime.date(year, 3, spring_day))
    holidays.add(datetime.date(year, 9, autumn_day))

    extra = set()
    for h in holidays:
        if h.weekday() == 6:
            extra.add(h + datetime.timedelta(days=1))
    holidays |= extra
    return holidays


def is_market_open_today():
    today = datetime.date.today()
    if today.weekday() >= 5:
        day_name = "土曜日" if today.weekday() == 5 else "日曜日"
        print(f"[スキップ] 本日（{today}）は{day_name}のため市場休場です。")
        return False
    if today in get_jp_holidays(today.year):
        print(f"[スキップ] 本日（{today}）は日本の祝日のため市場休場です。")
        return False
    return True


# ─── 設定読み込み ─────────────────────────────────────────────────
def load_config():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# ─── 前日データ ───────────────────────────────────────────────────
def load_previous():
    if PREV_FILE.exists():
        try:
            with open(PREV_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return {}
                return json.loads(content)
        except Exception:
            return {}
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

def calc_atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])

def fetch_indicators(ticker):
    for attempt in range(3):
        try:
            df = yf.download(ticker, period="6mo", progress=False, auto_adjust=True)
            if not df.empty:
                info     = yf.Ticker(ticker).info
                name_raw = info.get("longName") or info.get("shortName") or ticker
                name     = info.get("longNameJa") or info.get("shortNameJa") or name_raw
                close    = df["Close"].squeeze()
                high     = df["High"].squeeze()
                low      = df["Low"].squeeze()
                volume   = df["Volume"].squeeze()
                price      = float(close.iloc[-1])
                prev_price = float(close.iloc[-2])
                bb_upper, bb_mid, bb_lower = calc_bollinger(close)
                atr = calc_atr(high, low, close)
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
                    "atr":        atr,
                    "atr_pct":    atr / price * 100,
                    "volume":     float(volume.iloc[-1]),
                    "volume_ma5": float(volume.rolling(5).mean().iloc[-1]),
                }
                return ind
        except Exception as e:
            print(f"  [{ticker}] attempt {attempt+1} failed: {e}")
        time.sleep(2)
    return None

# ─── 観察開始日管理（連続日数カウント用） ────────────────────────
def get_observation_day(ticker, previous):
    """previous.jsonのcumulativeにstart_dateを持たせて日数を計算"""
    prev = previous.get(ticker, {})
    cum  = prev.get("cumulative", {})
    start_str = cum.get("start_date")
    today = datetime.date.today()
    if start_str:
        start = datetime.date.fromisoformat(start_str)
        # 取引日数ベースではなく暦日数ベース（シンプルに）
        delta = (today - start).days + 1
        return delta, start_str
    else:
        return 1, today.isoformat()

# ─── Claude 観察レポート生成 ──────────────────────────────────────
def generate_report(client, ind, prev_data):
    prev_ind  = prev_data.get("ind")  if prev_data else None
    prev_pred = prev_data.get("predictions") if prev_data else None

    if prev_ind:
        diff_text = f"""
【前日からの変化】
・株価: {prev_ind['price']:.2f} → {ind['price']:.2f}（{ind['change_pct']:+.2f}%）
・RSI: {prev_ind['rsi']:.1f} → {ind['rsi']:.1f}（{ind['rsi']-prev_ind['rsi']:+.1f}）
・MACDヒスト: {prev_ind['macd_hist']:.3f} → {ind['macd_hist']:.3f}
・ATR: {prev_ind.get('atr', 0):.2f} → {ind['atr']:.2f}（株価比 {ind['atr_pct']:.2f}%）
・出来高: {prev_ind['volume']:,.0f} → {ind['volume']:,.0f}（5日平均比: {ind['volume']/ind['volume_ma5']*100:.0f}%）
"""
        prev_score = prev_data.get("score")
        prev_score_text = f"前日の上昇期待度: {prev_score}点" if prev_score is not None else "前日スコア: なし"
    else:
        diff_text = "【前日データ】初回観察のため比較なし"
        prev_score_text = "前日スコア: なし（初回）"

    if prev_pred:
        bull               = prev_pred.get("bullish_price", 0)
        bear               = prev_pred.get("bearish_price", 0)
        neutral_range      = prev_pred.get("neutral_range", "")
        predicted_scenario = prev_pred.get("scenario", "")
        actual_scenario    = "上昇" if ind["change_pct"] >= 0.5 else ("下落" if ind["change_pct"] <= -0.5 else "横ばい")
        hit = "的中" if predicted_scenario == actual_scenario else "外れ"
        answer_text = f"""
【昨日の予測答え合わせ】
・予測シナリオ（最有力）: {predicted_scenario}
・予測価格帯: 上昇={bull:,.0f}円 / 横ばい={neutral_range}円 / 下落={bear:,.0f}円
・実際の結果: {actual_scenario}（{ind['price']:,.2f}円 / {ind['change_pct']:+.2f}%）
・判定: {hit}
"""
    else:
        answer_text = "【昨日の予測答え合わせ】初回観察のためなし"

    prompt = f"""あなたは株式マーケットの観察記録を担当するアナリストです。
投資推奨ではなく、銘柄の状態を客観的に記録・観察するレポートを日本語で書いてください。

【銘柄】{ind['ticker']}（{ind['name']}）
【現在値】{ind['price']:,.2f}円（前日比 {ind['change_pct']:+.2f}%）
【移動平均線】MA5={ind['ma5']:,.2f} / MA25={ind['ma25']:,.2f} / MA75={ind['ma75']:,.2f}
【RSI(14)】{ind['rsi']:.1f}
【MACD】ライン={ind['macd']:.3f} / シグナル={ind['macd_sig']:.3f} / ヒスト={ind['macd_hist']:.3f}（前日={ind['macd_prev']:.3f}）
【ボリンジャーバンド】上限={ind['bb_upper']:,.2f} / 中央={ind['bb_mid']:,.2f} / 下限={ind['bb_lower']:,.2f}
【ATR(14)】{ind['atr']:.2f}円（株価比 {ind['atr_pct']:.2f}%）
【出来高】直近={ind['volume']:,.0f} / 5日平均={ind['volume_ma5']:,.0f}（平均比 {ind['volume']/ind['volume_ma5']*100:.0f}%）
{diff_text}
{answer_text}
{prev_score_text}

以下の形式で出力してください。「エントリー」「利確」「損切り」などの売買用語は使わず、観察・記録の視点で書いてください。

---AI観察コメント---
（100〜150字で、今日の状態を一言で表す。MACDや出来高・ATRなど注目指標に触れること。ATRが高ければ値動きが荒い旨を、低ければ膠着状態を示唆する旨を含めること）

---昨日の予測を振り返って---
（前日予測がなぜ当たった／外れたかを指標の動きから考察。60〜100字）
※初回観察の場合はこのセクションを省略してください。

---上昇期待度---
以下の採点基準で合計点を計算し、数字のみ（0〜100の整数）を出力してください。

【採点基準（合計100点）】
■ トレンド方向（30点）
・株価がMA5・MA25・MA75すべて上回っている → 30点
・MA25・MA75のみ上回っている → 20点
・MA75のみ上回っている → 10点
・すべて下回っている → 0点

■ モメンタム＝MACD（25点）
・MACDヒストがプラスかつ前日より拡大 → 25点
・MACDヒストがプラスだが縮小 → 15点
・MACDヒストがマイナスだが縮小中（底打ち兆候） → 10点
・MACDヒストがマイナスかつ拡大（下落加速） → 0点

■ 過熱感＝RSI（20点）
・RSI 45〜65（健全な上昇圏） → 20点
・RSI 65〜75（やや過熱だが上昇余地あり） → 12点
・RSI 75以上（過熱圏） → 5点
・RSI 45未満（弱気圏） → 0点

■ 出来高（15点）
・直近出来高が5日平均の110%以上 → 15点
・90〜110%（平均並み） → 10点
・90%未満（低調） → 5点

■ ボリンジャーバンド（10点）
・株価がBBミッド〜BB上限の間 → 10点
・株価がBB上限を超えている（突破または過熱） → 5点
・株価がBBミッド未満 → 0点

合計点を0〜100の整数で出力。

---明日の予測---
（ATRを参考に価格レンジを算出すること。ATR値を±の目安として使用。）
上昇シナリオ: （価格）円
横ばいシナリオ: （価格レンジ）円
下落シナリオ: （価格）円
最有力シナリオ: 上昇 or 横ばい or 下落

---note見出し---
（その日の相場を表す10文字前後の見出しを1行で出力。句読点なし。例：膠着続く弱気の攻防）

---AI自己採点総括---
（前日スコアと本日スコアを比較し、「前日〇点→本日〇点」という変化の自己評価を50〜80字で記述。スコアが上昇した場合は改善要因を、下落した場合は悪化要因を客観的に述べること。）

---セクション終わり---"""

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text

# ─── レスポンスパース ─────────────────────────────────────────────
def parse_report(raw_text):
    result = {
        "comment":        "",
        "review":         "",
        "score":          None,
        "headline":       "",
        "self_eval":      "",
        "predictions": {
            "bullish_price": None,
            "neutral_range": None,
            "bearish_price": None,
            "scenario":      None,
        },
    }

    lines           = raw_text.splitlines()
    current_section = None

    for line in lines:
        stripped = line.strip()

        if "---AI観察コメント---" in stripped:
            current_section = "comment"; continue
        elif "---昨日の予測を振り返って---" in stripped:
            current_section = "review";  continue
        elif "---上昇期待度---" in stripped:
            current_section = "score";   continue
        elif "---明日の予測---" in stripped:
            current_section = "pred";    continue
        elif "---note見出し---" in stripped:
            current_section = "headline"; continue
        elif "---AI自己採点総括---" in stripped:
            current_section = "self_eval"; continue
        elif "---セクション終わり---" in stripped:
            current_section = None;      continue

        if current_section == "comment" and stripped:
            result["comment"] += stripped + " "
        elif current_section == "review" and stripped:
            result["review"] += stripped + " "
        elif current_section == "score" and stripped:
            try:
                result["score"] = int("".join(filter(str.isdigit, stripped)))
            except Exception:
                pass
        elif current_section == "pred" and stripped:
            if stripped.startswith("上昇シナリオ"):
                try:
                    result["predictions"]["bullish_price"] = int(
                        "".join(filter(str.isdigit, stripped.split(":")[-1]))
                    )
                except Exception:
                    pass
            elif stripped.startswith("横ばいシナリオ"):
                result["predictions"]["neutral_range"] = stripped.split(":")[-1].strip().replace("円", "")
            elif stripped.startswith("下落シナリオ"):
                try:
                    result["predictions"]["bearish_price"] = int(
                        "".join(filter(str.isdigit, stripped.split(":")[-1]))
                    )
                except Exception:
                    pass
            elif stripped.startswith("最有力シナリオ"):
                val = stripped.split(":")[-1].strip()
                for keyword in ["上昇", "横ばい", "下落"]:
                    if keyword in val:
                        result["predictions"]["scenario"] = keyword
                        break
        elif current_section == "headline" and stripped:
            result["headline"] = stripped
        elif current_section == "self_eval" and stripped:
            result["self_eval"] += stripped + " "

    result["comment"]   = result["comment"].strip()
    result["review"]    = result["review"].strip()
    result["headline"]  = result["headline"].strip()
    result["self_eval"] = result["self_eval"].strip()
    return result

# ─── 累計勝敗集計 ────────────────────────────────────────────────
def calc_cumulative_record(ticker, previous, current_hit):
    prev = previous.get(ticker, {})
    cum  = dict(prev.get("cumulative", {"win": 0, "lose": 0}))
    if current_hit is True:
        cum["win"] += 1
    elif current_hit is False:
        cum["lose"] += 1
    return cum["win"], cum["lose"]

# ─── note投稿用テキスト生成（1銘柄分） ───────────────────────────
def build_note_text_single(date_str, r, previous, obs_day):
    today  = datetime.date.today()
    ind    = r["ind"]
    parsed = r["parsed"]
    ticker = ind["ticker"]
    prev   = previous.get(ticker, {})
    pred   = prev.get("predictions") if prev else None
    prev_score = prev.get("score") if prev else None

    change_sign = "▲" if ind["change_pct"] >= 0 else "▼"
    change_abs  = abs(ind["change_pct"])
    vol_ratio   = ind["volume"] / ind["volume_ma5"] * 100
    actual      = "上昇" if ind["change_pct"] >= 0.5 else ("下落" if ind["change_pct"] <= -0.5 else "横ばい")

    if pred and pred.get("scenario"):
        current_hit = pred["scenario"] == actual
    else:
        current_hit = None

    score = parsed["score"]
    headline = parsed.get("headline", "")
    self_eval = parsed.get("self_eval", "")

    # ─── タイトル生成 ───────────────────────────────────────────
    # 【定点観測銘柄〇日目】サブタイトル / 見出し
    score_label = "強気" if (score or 0) >= 70 else ("中立" if (score or 0) >= 50 else "弱気")
    sub_title   = f"{ind['name']}（{ind['ticker']}）{obs_day}日目"
    note_title  = f"【定点観測銘柄{obs_day}日目】{headline}"

    lines = []
    lines.append("【noteタイトル候補】")
    lines.append(note_title)
    lines.append(f"サブタイトル：{sub_title}")
    lines.append("")
    lines.append("=" * 50)
    lines.append("【ここから本文をコピーしてnoteに貼ってください】")
    lines.append("=" * 50)
    lines.append("")
    # ★変更②：定型文の後に銘柄名・日数表記なし
    lines.append(f"本日も{ind['name']}をAIでテクニカル分析しました。前日予測の結果と合わせて確認しながら、チャート指標を中心にAIの市場分析精度を日々検証しています。")
    lines.append("")
    lines.append("━" * 30)
    lines.append(f"■ {ind['name']}（{ind['ticker']}）")
    lines.append("")
    lines.append("【本日の主要指標】")
    lines.append(f"現在値　：{ind['price']:,.0f}円（{change_sign}{change_abs:.2f}%）")
    lines.append(f"RSI　　 ：{ind['rsi']:.1f}")
    lines.append(f"MA5　　 ：{ind['ma5']:,.0f}円")
    lines.append(f"MA25　　：{ind['ma25']:,.0f}円")
    lines.append(f"MACD　　：{ind['macd_hist']:.3f}")
    lines.append(f"ATR　　 ：{ind['atr']:.2f}円（株価比 {ind['atr_pct']:.2f}%）")
    lines.append(f"出来高　：{ind['volume']:,.0f}（5日平均比 {vol_ratio:.0f}%）")
    lines.append("")
    lines.append("【AI観察コメント】")
    lines.append(parsed["comment"] if parsed["comment"] else "（取得できませんでした）")
    lines.append("")

    if pred and pred.get("scenario"):
        hit = "✅ 的中" if current_hit else "❌ 外れ"
        lines.append("【昨日の予測答え合わせ】")
        lines.append(f"昨日の予測：{pred['scenario']}　実際：{actual}　→ {hit}")
        if parsed["review"]:
            lines.append(f"振り返り：{parsed['review']}")
        lines.append("")

    # ★変更③：予測シナリオ → 明日の上昇期待度の順
    p = parsed["predictions"]
    if p["bullish_price"] or p["neutral_range"] or p["bearish_price"]:
        lines.append("【明日の予測シナリオ】")
        if p["bullish_price"]:
            lines.append(f"上昇　　：{p['bullish_price']:,}円")
        if p["neutral_range"]:
            lines.append(f"横ばい　：{p['neutral_range']}円")
        if p["bearish_price"]:
            lines.append(f"下落　　：{p['bearish_price']:,}円")
        lines.append("")

    # ★変更③：「明日の上昇期待度」として予測シナリオ直後に表示
    if score is not None:
        score_str = f"{score}点 / 100点（{score_label}）"
        lines.append(f"明日の上昇期待度：{score_str}")
        lines.append("※トレンド・MACD・RSI・出来高・BBを採点した総合スコアです")
        lines.append("")

    # ★変更④：【総括】 ①前日→本日スコア ②AI自己採点 ③コメント、勝敗廃止
    lines.append("【総括】")
    # ①前日→本日スコア変化
    if prev_score is not None and score is not None:
        direction = "改善" if score > prev_score else ("悪化" if score < prev_score else "横ばい")
        lines.append(f"前日の上昇期待度{prev_score}点に対し本日は{score}点（{direction}）")
    elif score is not None:
        lines.append(f"本日の上昇期待度：{score}点（初回観察）")
    # ②AI自己採点
    if self_eval:
        lines.append(f"AI自己採点：{self_eval}")
    # ③本日の動き総括コメント
    lines.append(f"本日{actual}・{score_label}圏で推移")
    lines.append("")
    lines.append("━" * 30)
    lines.append("※本記事はAIによる市場観察記録であり、投資助言を目的とするものではありません。")
    lines.append("")
    # ★変更⑤：ハッシュタグにUFJ・ソニー追加
    lines.append(f"#株式観察 #テクニカル分析 #定点観測 #AI予測検証 #UFJ #ソニー #{today.strftime('%Y%m%d')}")

    return "\n".join(lines)

# ─── HTML生成 ─────────────────────────────────────────────────────
def build_report_html(date_str, results, previous):
    cards = ""
    for r in results:
        ind    = r["ind"]
        parsed = r["parsed"]
        ticker = ind["ticker"]
        prev   = previous.get(ticker, {})
        pred   = prev.get("predictions") if prev else None

        change_class = "up" if ind["change_pct"] >= 0 else "down"
        change_sign  = "+" if ind["change_pct"] >= 0 else ""
        vol_ratio    = ind["volume"] / ind["volume_ma5"] * 100
        actual       = "上昇" if ind["change_pct"] >= 0.5 else ("下落" if ind["change_pct"] <= -0.5 else "横ばい")

        if pred and pred.get("scenario"):
            current_hit = pred["scenario"] == actual
        else:
            current_hit = None

        cum_win, cum_lose = calc_cumulative_record(ticker, previous, current_hit)
        cum_total = cum_win + cum_lose
        cum_rate  = f"{int(cum_win/cum_total*100)}%" if cum_total > 0 else "―"

        if pred and pred.get("scenario"):
            hit_cls = "hit" if current_hit else "miss"
            hit_str = "✅ 的中" if current_hit else "❌ 外れ"
            review_html = f'<p class="review">{parsed["review"]}</p>' if parsed["review"] else ""
            answer_html = f"""
          <div class="answer-box {hit_cls}">
            <span class="answer-label">昨日の予測答え合わせ</span>
            <span class="answer-result">{hit_str}</span>
            <span class="answer-detail">予測：{pred['scenario']} → 実際：{actual}</span>
            {review_html}
          </div>"""
        else:
            answer_html = ""

        p = parsed["predictions"]
        score = parsed["score"]
        score_str = f"{score}点" if score is not None else "―"

        # ★HTML版も予測シナリオ → 明日の上昇期待度の順
        pred_html = ""
        if p["bullish_price"] or p["neutral_range"] or p["bearish_price"]:
            pred_html = f"""
          <div class="pred-box">
            <span class="pred-label">明日の予測シナリオ（ATR基準）</span>
            <div class="pred-scenarios">
              <span class="scenario up-s">上昇 {p['bullish_price']:,}円</span>
              <span class="scenario neu-s">横ばい {p['neutral_range']}円</span>
              <span class="scenario down-s">下落 {p['bearish_price']:,}円</span>
            </div>
          </div>
          <div class="score-badge" style="margin-top:0.8rem">明日の上昇期待度 <strong>{score_str}</strong></div>"""

        # ★HTML版総括
        prev_score = prev.get("score")
        self_eval  = parsed.get("self_eval", "")
        if prev_score is not None and score is not None:
            direction = "改善" if score > prev_score else ("悪化" if score < prev_score else "横ばい")
            summary_line = f"前日{prev_score}点 → 本日{score}点（{direction}）"
        elif score is not None:
            summary_line = f"本日{score}点（初回）"
        else:
            summary_line = ""

        summary_html = ""
        if summary_line or self_eval:
            summary_html = f"""
          <div class="summary-box">
            <span class="summary-label">総括</span>
            {f'<p class="summary-score">{summary_line}</p>' if summary_line else ""}
            {f'<p class="summary-eval">{self_eval}</p>' if self_eval else ""}
          </div>"""

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
            <div class="metric"><span class="label">MA5</span><span class="value">{ind['ma5']:,.0f}</span></div>
            <div class="metric"><span class="label">MA25</span><span class="value">{ind['ma25']:,.0f}</span></div>
            <div class="metric"><span class="label">MACDヒスト</span><span class="value">{ind['macd_hist']:.3f}</span></div>
            <div class="metric"><span class="label">ATR</span><span class="value">{ind['atr']:.2f}円</span></div>
            <div class="metric"><span class="label">出来高比</span><span class="value">{vol_ratio:.0f}%</span></div>
          </div>
          <div class="report-body">
            <p class="comment">{parsed['comment']}</p>
            {answer_html}
            {pred_html}
            {summary_html}
          </div>
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
    .metrics {{ display: flex; gap: 0; border-bottom: 1px solid var(--border); flex-wrap: wrap; }}
    .metric {{ flex: 1; min-width: 80px; padding: 0.8rem 1rem; border-right: 1px solid var(--border); text-align: center; }}
    .metric:last-child {{ border-right: none; }}
    .metric .label {{ display: block; font-size: 0.7rem; color: var(--muted); letter-spacing: 0.05em; margin-bottom: 0.3rem; }}
    .metric .value {{ font-family: 'JetBrains Mono', monospace; font-size: 0.95rem; font-weight: 600; }}
    .report-body {{ padding: 1.5rem; font-size: 0.92rem; color: #cdd9e5; }}
    .score-badge {{ display: inline-block; background: #21262d; border: 1px solid var(--border); border-radius: 6px; padding: 0.3rem 0.8rem; font-size: 0.85rem; margin-bottom: 0.8rem; }}
    .score-badge strong {{ color: var(--accent); font-size: 1.05rem; }}
    .comment {{ line-height: 1.8; margin-bottom: 1rem; }}
    .answer-box {{ border-radius: 6px; padding: 0.8rem 1rem; margin-bottom: 1rem; }}
    .answer-box.hit {{ background: rgba(63,185,80,0.1); border: 1px solid rgba(63,185,80,0.3); }}
    .answer-box.miss {{ background: rgba(248,81,73,0.1); border: 1px solid rgba(248,81,73,0.3); }}
    .answer-label {{ display: block; font-size: 0.75rem; color: var(--muted); margin-bottom: 0.3rem; }}
    .answer-result {{ font-size: 1rem; font-weight: 600; margin-right: 0.8rem; }}
    .answer-detail {{ font-size: 0.85rem; color: var(--muted); }}
    .review {{ font-size: 0.85rem; color: #adbac7; margin-top: 0.5rem; line-height: 1.7; }}
    .pred-box {{ background: #21262d; border-radius: 6px; padding: 0.8rem 1rem; margin-bottom: 0.8rem; }}
    .pred-label {{ display: block; font-size: 0.75rem; color: var(--muted); margin-bottom: 0.5rem; }}
    .pred-scenarios {{ display: flex; gap: 0.6rem; flex-wrap: wrap; }}
    .scenario {{ font-size: 0.85rem; padding: 0.3rem 0.7rem; border-radius: 4px; font-family: 'JetBrains Mono', monospace; }}
    .up-s {{ background: rgba(63,185,80,0.15); color: var(--up); }}
    .neu-s {{ background: rgba(88,166,255,0.15); color: var(--accent); }}
    .down-s {{ background: rgba(248,81,73,0.15); color: var(--down); }}
    .summary-box {{ background: #21262d; border-radius: 6px; padding: 0.8rem 1rem; margin-top: 0.8rem; border-left: 3px solid var(--accent); }}
    .summary-label {{ display: block; font-size: 0.75rem; color: var(--muted); margin-bottom: 0.4rem; }}
    .summary-score {{ font-size: 0.9rem; font-weight: 600; margin-bottom: 0.3rem; }}
    .summary-eval {{ font-size: 0.85rem; color: #adbac7; line-height: 1.7; }}
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
    if not is_market_open_today():
        print("本日は市場休場のため処理を終了します。")
        sys.exit(0)

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

        prev_data = previous.get(ticker)
        print(f"[{ticker}] Claudeでレポート生成中...")
        raw_report = generate_report(client, ind, prev_data)
        parsed     = parse_report(raw_report)

        # 本日の的中判定（累計用）
        prev_pred = prev_data.get("predictions") if prev_data else None
        if prev_pred and prev_pred.get("scenario"):
            actual      = "上昇" if ind["change_pct"] >= 0.5 else ("下落" if ind["change_pct"] <= -0.5 else "横ばい")
            current_hit = prev_pred["scenario"] == actual
        else:
            current_hit = None

        cum_win, cum_lose = calc_cumulative_record(ticker, previous, current_hit)

        # 観察開始日を初回のみ記録
        prev_cum   = previous.get(ticker, {}).get("cumulative", {})
        start_date = prev_cum.get("start_date", today.isoformat())

        new_prev[ticker] = {
            "ind":         ind,
            "predictions": parsed["predictions"],
            "score":       parsed["score"],   # ★スコアを保存
            "cumulative":  {
                "win":        cum_win,
                "lose":       cum_lose,
                "start_date": start_date,
            },
        }

        results.append({"ind": ind, "report": raw_report, "parsed": parsed})
        time.sleep(1)

    # レポートHTML保存
    report_html = build_report_html(date_str, results, previous)
    report_path = REPORT_DIR / fname
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_html)
    print(f"レポートHTML保存: {report_path}")

    # note投稿用テキスト保存（銘柄ごとに1ファイル）
    for r in results:
        ticker      = r["ind"]["ticker"]
        ticker_safe = ticker.replace(".", "-")
        obs_day, _  = get_observation_day(ticker, previous)
        note_text   = build_note_text_single(date_str, r, previous, obs_day)
        note_path   = NOTE_DIR / f"{today.strftime('%Y-%m-%d')}_{ticker_safe}.txt"
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
