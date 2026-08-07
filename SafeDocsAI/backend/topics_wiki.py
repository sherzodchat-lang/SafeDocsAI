#!/usr/bin/env python3
"""Статьи Википедии по предметам — прямо, без агента.

Зачем свой скрипт, если сбор поручался агентам. Три из пяти справились, два
застряли: час правили собственные скрипты и не собрали ни одной записи. Предметы
у них были самые нужные — медицина, образование, наука, техника, связь,
транспорт, энергетика, — то есть ровно те, ради которых всё затевалось: без них
законы об энергетике и о медицине останутся в одной куче «Законодательство».

Задача при этом простая и уже решённая теми тремя: у Википедии есть API, он
отдаёт чистый текст и номер ревизии, и ходить по нему надо через категории.
Писать это самому быстрее, чем объяснять ещё раз.

КАК ПОДБИРАЮТСЯ СТАТЬИ. Категории надёжнее поиска: поиск по слову «энергия»
принесёт «Энергию (журнал)» и «Жизненную энергию», а категория «Энергетика» —
статьи об энергетике. Обход идёт вглубь на два уровня подкатегорий, потому что
верхний уровень часто состоит из одних подкатегорий и почти не содержит статей.

ЯЗЫКИ. Берём сколько есть по-таджикски и добираем русскими. Таджикская
Википедия маленькая, требовать от неё по 25 статей на предмет — значит не
собрать ничего; ровно на этом и застряли агенты первого захода.

    ./venv/bin/python topics_wiki.py --out data/topics_dataset/wiki_ab.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

# Википедия прямо требует называть себя и цель: безымянные запросы она вправе
# отклонять, и это не формальность, а условие пользования API.
UA = "SafeDocsAI-research/1.0 (academic topic-model training; contact via project repo)"

# Пауза между запросами. API Википедии жёсткого предела не ставит, но просит
# быть умеренным; полсекунды — та умеренность, при которой сбор всё равно
# укладывается в минуты.
PAUSE = 0.4

MIN_CHARS = 2500
TARGET_PER_SUBJECT = 30
CATEGORY_DEPTH = 2

# Предмет -> категории, с которых начинать обход. По две-три на язык: одна
# категория часто оказывается вырожденной (пустой или из одних подкатегорий), и
# запасная спасает предмет целиком.
SUBJECTS: dict[str, dict[str, list[str]]] = {
    "медицина и здравоохранение": {
        "ru": ["Медицина", "Здравоохранение", "Заболевания по алфавиту"],
        "tg": ["Тиб", "Тандурустӣ"],
    },
    "образование": {
        "ru": ["Образование", "Педагогика", "Высшее образование"],
        "tg": ["Маориф", "Таҳсилот"],
    },
    "наука": {
        "ru": ["Наука", "Научные дисциплины", "История науки"],
        "tg": ["Илм"],
    },
    "биология": {
        "ru": ["Биология", "Ботаника", "Зоология"],
        "tg": ["Биология", "Наботот"],
    },
    "психология и педагогика": {
        "ru": ["Психология", "Педагогика", "Психологические теории"],
        "tg": ["Психология"],
    },
    "языкознание": {
        "ru": ["Лингвистика", "Языки мира", "Грамматика"],
        "tg": ["Забоншиносӣ", "Забонҳо"],
    },
    "философия": {
        "ru": ["Философия", "Философские направления", "Философы"],
        "tg": ["Фалсафа"],
    },
    "техника и технологии": {
        "ru": ["Техника", "Технологии", "Машиностроение"],
        "tg": ["Техника", "Технология"],
    },
    "информатика и связь": {
        "ru": ["Информатика", "Телекоммуникации", "Интернет"],
        "tg": ["Информатика", "Алоқа"],
    },
    "транспорт": {
        "ru": ["Транспорт", "Железнодорожный транспорт", "Автомобильный транспорт"],
        "tg": ["Нақлиёт"],
    },
    "энергетика": {
        "ru": ["Энергетика", "Электроэнергетика", "Возобновляемая энергетика"],
        "tg": ["Энергетика", "Нерӯгоҳ"],
    },
    "строительство и архитектура": {
        "ru": ["Строительство", "Архитектура", "Здания и сооружения"],
        "tg": ["Меъморӣ", "Сохтмон"],
    },
    "физика и химия": {
        "ru": ["Физика", "Химия", "Физические величины"],
        "tg": ["Физика", "Химия"],
    },
    "математика": {
        "ru": ["Математика", "Разделы математики", "Математический анализ"],
        "tg": ["Математика"],
    },
}


class Wiki:
    def __init__(self, lang: str) -> None:
        self.url = f"https://{lang}.wikipedia.org/w/api.php"
        self.lang = lang
        self.session = requests.Session()
        self.session.headers["User-Agent"] = UA
        self.last = 0.0

    def call(self, **params) -> dict:
        wait = PAUSE - (time.monotonic() - self.last)
        if wait > 0:
            time.sleep(wait)
        params.update(format="json", formatversion=2)
        for attempt in range(4):
            try:
                response = self.session.get(self.url, params=params, timeout=60)
            except requests.RequestException:
                time.sleep(2 * (attempt + 1))
                continue
            self.last = time.monotonic()
            if response.status_code == 429:
                # Просят сбавить — сбавляем, а не долбим.
                time.sleep(10 * (attempt + 1))
                continue
            if response.ok:
                return response.json()
            time.sleep(2 * (attempt + 1))
        return {}

    def category_pages(self, category: str, depth: int = CATEGORY_DEPTH) -> list[int]:
        """Идентификаторы статей категории и её подкатегорий.

        Вглубь, потому что верхний уровень часто состоит из одних подкатегорий:
        у «Энергетики» прямых статей единицы, а у её подкатегорий — сотни.
        """
        pages: list[int] = []
        frontier = [(category, 0)]
        seen: set[str] = set()
        while frontier:
            name, level = frontier.pop(0)
            if name in seen:
                continue
            seen.add(name)
            payload = self.call(
                action="query",
                list="categorymembers",
                cmtitle=f"Category:{name}",
                cmlimit=200,
                cmtype="page|subcat",
                # Только статьи и категории: без этого в выборку лезут
                # «Портал:…» и «Шаблон:…», а это не документы.
                cmnamespace="0|14",
            )
            for member in (payload.get("query") or {}).get("categorymembers", []):
                title = member.get("title", "")
                if title.startswith(("Category:", "Категория:", "Гурӯҳ:")):
                    if level < depth:
                        frontier.append((title.split(":", 1)[1], level + 1))
                elif member.get("pageid"):
                    pages.append(int(member["pageid"]))
            if len(pages) > 600:
                break
        return pages

    def extracts(self, page_ids: list[int]) -> list[dict]:
        """Тексты статей — ПО ОДНОЙ, и оба слова здесь важны.

        Две ошибки, из-за которых первая версия этого файла не собирала ничего,
        и обе невидимы: запросы уходили, ответы приходили, записей не
        появлялось.

        ПЕРВАЯ: `exintro=0`. В API MediaWiki булев параметр истинен уже тем, что
        передан, — значение не смотрят. То есть «exintro=0» означает «только
        вступление», а не «не только вступление». Тексты приходили по
        полторы-две тысячи знаков и все до одного отсеивались порогом
        MIN_CHARS = 2500.

        ВТОРАЯ: пачки. Полные тексты API отдаёт по одной статье на запрос и
        честно об этом предупреждает («exlimit was too large for a whole article
        extracts request, lowered to 1»), но предупреждение лежит в поле,
        которого никто не читает: в пачке текст получала первая страница, а
        остальные приходили пустыми.

        Медленнее, чем пачками, — но пачками не работает вовсе.
        """
        out: list[dict] = []
        for page_id in page_ids:
            payload = self.call(
                action="query",
                pageids=str(page_id),
                prop="extracts|info",
                explaintext=1,
                inprop="url",
            )
            for page in (payload.get("query") or {}).get("pages", []):
                text = (page.get("extract") or "").strip()
                if len(text) < MIN_CHARS:
                    continue
                out.append(
                    {
                        "id": f"wiki-{self.lang}-{page['pageid']}",
                        "source": f"{self.lang}.wikipedia.org",
                        "url": page.get("fullurl", ""),
                        "title": page.get("title", ""),
                        "text": text,
                        "lang": "tj" if self.lang == "tg" else self.lang,
                        "published_at": str(page.get("lastrevid", "")),
                    }
                )
        return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data/topics_dataset/wiki_ab.jsonl")
    parser.add_argument("--target", type=int, default=TARGET_PER_SUBJECT)
    args = parser.parse_args()

    clients = {"tg": Wiki("tg"), "ru": Wiki("ru")}
    seen_urls: set[str] = set()
    rows: list[dict] = []

    for subject, categories in SUBJECTS.items():
        collected: list[dict] = []
        # Таджикский первым: его мало, и добирать русскими надо ровно столько,
        # сколько не хватило.
        for lang in ("tg", "ru"):
            for category in categories.get(lang, []):
                if len(collected) >= args.target:
                    break
                ids = clients[lang].category_pages(category)
                if not ids:
                    continue
                need = args.target - len(collected)
                for row in clients[lang].extracts(ids[: need * 4]):
                    if row["url"] in seen_urls:
                        continue
                    seen_urls.add(row["url"])
                    row["section"] = subject
                    collected.append(row)
                    if len(collected) >= args.target:
                        break
        rows.extend(collected)
        langs = {}
        for row in collected:
            langs[row["lang"]] = langs.get(row["lang"], 0) + 1
        print(f"  {len(collected):3d}  {subject:32s} {langs}", flush=True)

    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nзаписано {len(rows)} статей в {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
