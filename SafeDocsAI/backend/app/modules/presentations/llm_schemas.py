"""Схемы ответов модели для генерации презентаций и их разбор.

Здесь живут три вещи: форма плана, форма слайда и разбор сырого ответа модели.
Модуль заведён на этапе 0 (прототип) и на этапе 1 переехал в пайплайн без
переписывания — правки касаются только границ и нормализации.

Форма слайда — РАЗМЕЧЕННОЕ ОБЪЕДИНЕНИЕ по полю layout (см. SLIDE_LAYOUTS и
PresentationSlide ниже). До этого вид слайда был один — {heading, bullets,
citations}, — и колода получалась однообразной по самой простой причине: у
модели не было слова, которым говорят «здесь сравнение», «здесь одна цифра»,
«здесь этапы». Инвариант при этом не изменился ни на грамм: модель по-прежнему
пишет ТОЛЬКО поля схемы, а рисует их код. Раскладка — это выбор из пяти
закрытых значений, а не разметка.

ВЫБИРАЕТ раскладку при этом план (PlanSection.layout), а не слайд: форму
материала видно только тогда, когда виден весь материал. Слайд-вызов
назначенное исполняет, и потому одно и то же поле layout стоит в двух схемах —
в секции плана как задание и в слайде как отчёт о выполнении.

Схема — не только проверка, но и НОРМАЛИЗАТОР: приведение chunk_id к строке
(см. SlideCitation) и схлопывание повторяющихся цитат
(см. PresentationSlide._deduplicate_citations) объявлены частью контракта и
сделаны в одном месте, чтобы ни одному потребителю ниже по течению не
приходилось повторять их у себя.

Главное правило модуля — «цитата обязана указывать на чанк, который модели
реально показали». Проверка держится не на доверии к модели, а на множестве
chunk_id, собранном при сборке промпта: ответ со ссылкой на чужой (или
выдуманный) чанк считается невалидным целиком. Побочный, но важный эффект —
инъекция через описание презентации теряет смысл: даже уговорив модель
сослаться на несуществующий документ, атакующий получит отказ валидатора, а не
слайд с фальшивым источником.
"""

import json
import re
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    model_validator,
)

# Титульный слайд и финальный «Источники» рисует рендерер, а не модель.
# slide_count во всём модуле — ИТОГОВОЕ число слайдов в файле, поэтому
# контентных секций в плане ровно на два меньше. Держать это соглашение в одном
# месте важнее, чем сэкономить константу: разойдясь, план и рендерер дают файл
# не той длины, которую заказал пользователь, и молча.
RENDERER_ADDED_SLIDES = 2
# Минимум — титул, один контентный слайд и источники.
MIN_SLIDE_COUNT = RENDERER_ADDED_SLIDES + 1

PLAN_TITLE_MAX_CHARS = 120
SECTION_HEADING_MAX_CHARS = 80
# Поисковый запрос секции уходит в тот же ретривал, что и вопрос чата, а тот
# рассчитан на фразу, а не на абзац: длинный запрос размывает эмбеддинг и
# тянет за собой лексический шум.
SECTION_SEARCH_QUERY_MAX_CHARS = 200

SLIDE_HEADING_MAX_CHARS = 80
SLIDE_BULLET_MAX_CHARS = 200
# Нижняя граница — 2, а не 3.
#
# Этап 0 показал, что слайды повторяют одни и те же факты. Причина двойная:
# на маленьком корпусе ретривал отдаёт всем секциям почти одно и то же, НО и
# модель добивает слайд до нижней границы, когда сказать больше нечего.
# Требование «минимум три буллета» — это прямое указание сочинить третий:
# схема не оставляет модели законного способа признать материал исчерпанным.
# Двойка такой способ даёт, а вместе с правилом в промпте слайда («если новых
# фактов нет, дай два буллета») превращает молчание из нарушения в ответ.
# Больший корпус второй половины проблемы не лечит — он её маскирует.
SLIDE_BULLETS_MIN = 2
SLIDE_BULLETS_MAX = 5

# --- Раскладки слайда ----------------------------------------------------
#
# ЗАКРЫТЫЙ список из пяти значений. До него у слайда был ровно один вид —
# {heading, bullets, citations}, — и модель физически не могла сказать «здесь
# сравнение», «здесь одна цифра», «здесь этапы»: в схеме не было слова, которым
# это говорится. Колода поэтому получалась однообразной не потому, что модель
# не видела разницы в материале, а потому, что разницу негде было записать.
#
# Список именно закрытый: раскладку рисует КОД, и раскладке, которой нет в
# шаблонах, взяться на слайде неоткуда. Открытый список означал бы, что модель
# однажды придумает шестую и слайд не отрисуется вовсе.
LAYOUT_BULLETS = "bullets"
LAYOUT_COMPARE = "compare"
LAYOUT_METRIC = "metric"
LAYOUT_STEPS = "steps"
LAYOUT_QUOTE = "quote"
SLIDE_LAYOUTS = (
    LAYOUT_BULLETS,
    LAYOUT_COMPARE,
    LAYOUT_METRIC,
    LAYOUT_STEPS,
    LAYOUT_QUOTE,
)

# Пределы раскладок. Каждое число — либо ВЫВЕДЕНО из уже существующего предела
# (тогда оно записано формулой, чтобы не разъехаться при правке исходного),
# либо объяснено физикой слайда: сколько знаков помещается в это место листа
# тем кеглем, которым это место набирается. Назначенных «на глаз» чисел здесь
# нет; проверка бюджета промпта на них тоже опирается (см. constants.py и
# tests/test_presentation_prompt_budget.py, где худший случай считается по
# КАЖДОМУ назначению плана: назначенная раскладка плюс bullets — единственная
# законная замена, — а не по одной раскладке за всех).

# Колонка сравнения — ровно половина ширины слайда, поэтому и заголовок, и
# строка в ней вдвое короче полноширинных. Записано делением, а не числом:
# поправят предел заголовка слайда — колонка поедет следом сама.
SLIDE_COMPARE_HEADING_MAX_CHARS = SLIDE_HEADING_MAX_CHARS // 2
SLIDE_COMPARE_BULLET_MAX_CHARS = SLIDE_BULLET_MAX_CHARS // 2
# Нижняя граница — та же двойка и по той же причине, что у обычного слайда
# (см. SLIDE_BULLETS_MIN): требование «минимум три» — это прямое указание
# сочинить третий. Верхняя строже пятёрки: сторон две, и 4 + 4 = 8 строк
# половинной длины — это уже та же масса текста, что 5 полноширинных, дальше
# слайд перестаёт читаться как сравнение и становится двумя списками.
SLIDE_COMPARE_BULLETS_MIN = SLIDE_BULLETS_MIN
SLIDE_COMPARE_BULLETS_MAX = 4

# Величина на слайде-цифре набирается кеглем во весь слайд — это единственное,
# ради чего слайд существует. 24 знака — предел, при котором «3,2 млрд сомонӣ»
# (15) и «12,5 % к 2030 году» (18) ещё влезают в одну строку такого кегля;
# всё, что длиннее, — уже не величина, а предложение, и ему место в caption.
SLIDE_METRIC_VALUE_MAX_CHARS = 24
# Подпись — фраза под числом, набранная как заголовок, но в две строки:
# «Доля электронной отчётности среди юридических лиц». 120 знаков ≈ 18 слов.
SLIDE_METRIC_CAPTION_MAX_CHARS = 120
# Уточнение под подписью — ровно один буллет по объёму, им и записано: место
# под него на слайде такое же, как под строку обычного списка.
SLIDE_METRIC_NOTE_MAX_CHARS = SLIDE_BULLET_MAX_CHARS

# Шагов от трёх до пяти. Два шага — это не процесс, а «до и после», то есть
# compare; шесть карточек в ряд на слайде уже нечитаемы — их пришлось бы
# набирать кеглем сноски.
SLIDE_STEPS_MIN = 3
SLIDE_STEPS_MAX = 5
# Заголовок шага живёт в карточке, а карточка — пятая часть ширины слайда.
# Он обязан лечь в одну строку, поэтому короче заголовка слайда.
SLIDE_STEP_TITLE_MAX_CHARS = 60
# Текст шага — та же строка, что и буллет, но внутри карточки, то есть с
# полями по бокам: 160 против 200.
SLIDE_STEP_TEXT_MAX_CHARS = 160

# Цитата занимает слайд целиком и набирается крупно — это два буллета текста,
# так и записано. Больше на слайд не влезет физически, а цитата длиннее двух
# фраз перестаёт быть цитатой и становится пересказом куска документа.
SLIDE_QUOTE_TEXT_MAX_CHARS = 2 * SLIDE_BULLET_MAX_CHARS
# Атрибуция — «Налоговый кодекс, статья 12», а не имя файла: имена документов
# рисует слайд «Источники». Тот же объём, что у подписи к величине.
SLIDE_QUOTE_ATTRIBUTION_MAX_CHARS = 120

_CODE_FENCE_RE = re.compile(
    r"```[ \t]*[a-zA-Z0-9_+-]*[ \t]*\r?\n(?P<body>.*?)(?:\r?\n)?[ \t]*```",
    re.DOTALL,
)

# Управляющие символы, которые модель ставит осмысленно: перевод строки внутри
# буллета и табуляция внутри текста. Их не выкидываем, а приводим к тому, чем
# они и должны были быть в JSON, — к экранированной форме. Остальным C0
# (\x00-\x1f: ESC, \x0c, обрывки CR и прочее) в тексте слайда делать нечего:
# они не несут смысла и доезжают до рендерера, где превращаются в видимый мусор
# вида "_x001B_" прямо на слайде.
_MEANINGFUL_CONTROL_ESCAPES = {"\n": "\\n", "\t": "\\t"}


def content_section_count(slide_count: int) -> int:
    """Сколько контентных секций должен вернуть план для slide_count слайдов."""
    return slide_count - RENDERER_ADDED_SLIDES


class LlmResponseError(Exception):
    """Ответ модели не разобран или не прошёл валидацию.

    error_text — текст, пригодный для подстановки в повторный промпт: он
    описывает, ЧТО именно не так, а не «ошибка разбора». Это единственная
    подсказка, которую модель получает на второй попытке, поэтому она обязана
    быть предметной.
    """

    def __init__(self, error_text: str) -> None:
        super().__init__(error_text)
        self.error_text = error_text


class PlanSection(BaseModel):
    """Секция плана: о чём слайд, чем его искать и КАКОЙ ОН БУДЕТ ФОРМЫ.

    layout стоит здесь, а не остаётся выбором слайд-вызова, по одной причине:
    выбрать форму можно только увидев материал целиком. Слайд-вызов видит свою
    секцию и больше ничего, и в такой позиции список — безопасный ответ на любой
    материал: он подходит всегда. Живая проверка это и показала — из восьми
    содержательных слайдов нестандартную раскладку получил один. Причина не в
    словах промпта: секция, уже сформулированная как перечисление, сравнением не
    станет, сколько её ни проси.

    План же видит дайджест корпуса, описание заказа и всю длину колоды сразу,
    поэтому способен сказать «здесь сравнение двух периодов», «здесь ключевая
    цифра», «здесь этапы реформы» — то есть выбрать форму ДО того, как секция
    сформулирована.

    Поле обязательное и без умолчания. Молчаливый фолбэк в bullets вернул бы нас
    ровно туда, откуда волна началась, и сделал бы это незаметно: колода
    собралась бы, просто снова однообразной.
    """

    heading: str = Field(min_length=1, max_length=SECTION_HEADING_MAX_CHARS)
    search_query: str = Field(min_length=1, max_length=SECTION_SEARCH_QUERY_MAX_CHARS)
    # Тот же закрытый список, что и у слайда, и записан теми же константами:
    # разъехавшись, план назначал бы раскладку, которой слайд-схема не знает, и
    # каждый такой слайд сгорал бы в повторной попытке.
    layout: Literal[
        LAYOUT_BULLETS,
        LAYOUT_COMPARE,
        LAYOUT_METRIC,
        LAYOUT_STEPS,
        LAYOUT_QUOTE,
    ]


class PresentationPlan(BaseModel):
    title: str = Field(min_length=1, max_length=PLAN_TITLE_MAX_CHARS)
    sections: list[PlanSection] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_section_count(self, info: ValidationInfo) -> "PresentationPlan":
        # Ожидаемое число секций зависит от запроса пользователя, а не от схемы,
        # поэтому приходит контекстом валидации: model_validate(..., context=...).
        # Без контекста проверка не выполняется — так схемой можно пользоваться
        # и там, где число слайдов ещё не известно (например, в тестах формы).
        expected = (info.context or {}).get("expected_sections")
        if expected is None:
            return self
        if len(self.sections) != expected:
            raise ValueError(
                f"sections must contain exactly {expected} items, got {len(self.sections)}"
            )
        return self


class SlideCitation(BaseModel):
    """Ссылка слайда на фрагмент.

    chunk_id объявлен контрактом как «ЧИСЛО ИЛИ СТРОКА, канонизируется в
    строку». Это не молчаливая подмена значения, а детерминированное
    приведение типа того же значения: 45 -> "45", без потерь и без выбора.

    Приведение узаконено, а не спрятано, по трём причинам. Первая: в системе
    chunk_id — строка (идентификатором вектора в ChromaDB служит str(chunk.id)),
    а модель почти всегда отдаёт его числом, потому что видит в промпте цифры;
    отвергать из-за формата целый слайд нечестно. Вторая: приведение живёт
    ровно в ОДНОМ месте — здесь, в схеме, — поэтому ни один потребитель ниже по
    течению не обязан помнить про два вида chunk_id и городить str() у себя.
    Третья: проверка на подмножество выданных chunk_id выполняется уже ПОСЛЕ
    канонизации (mode="before" отрабатывает раньше валидаторов модели), то есть
    от приведения она не слабеет ни на грамм.
    """

    source_id: int
    chunk_id: str

    @model_validator(mode="before")
    @classmethod
    def _canonicalize_chunk_id(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        value = data.get("chunk_id")
        # bool — подкласс int, но "True" идентификатором фрагмента не бывает:
        # такое значение обязано дойти до обычной ошибки типа, а не стать
        # строкой.
        if isinstance(value, bool) or not isinstance(value, int):
            return data
        return {**data, "chunk_id": str(value)}


def _reject_blank(value: str) -> str:
    """Строка из одних пробелов — не текст, а пустое место на слайде.

    min_length=1 её пропускает, а рендерер честно нарисует раскладку без
    содержимого: величину без величины, цитату без цитаты. Молчаливая обрезка
    здесь была бы хуже отказа — пустое поле означает, что модель не нашла
    материала под выбранную раскладку, и лечится это другой раскладкой, а не
    пробелом.
    """
    if not value.strip():
        raise ValueError("must not be blank")
    return value


def _bounded_text(max_chars: int) -> Any:
    """Тип текстового поля раскладки: непустое, не пробельное, не длиннее предела.

    Отдельный конструктор, а не Field в каждом объявлении, — чтобы «непустое и
    не из пробелов» не пришлось помнить в четырнадцати местах: одно забытое
    место означает раскладку, которую можно отрисовать пустой.
    """
    return Annotated[
        str,
        Field(min_length=1, max_length=max_chars),
        AfterValidator(_reject_blank),
    ]


def _check_bullet_list(bullets: list[str], *, layout: str, max_chars: int) -> None:
    """Проверить список строк слайда: непустые и не длиннее предела.

    Своё сообщение вместо Field-ограничения на элементе списка — ради НОМЕРА
    строки и её настоящей длины: «сократите bullets[3], в нём 240 знаков из
    200» модель исполняет, а «String should have at most 200 characters» без
    номера заставляет её сокращать наугад, чаще всего не ту строку.

    Одна функция на две раскладки (bullets и колонка compare), потому что
    предел у них разный, а правило одно: разъехавшись, две копии дали бы разные
    формулировки отказа на одну и ту же ошибку.
    """
    for index, bullet in enumerate(bullets):
        if not bullet.strip():
            raise ValueError(f"layout={layout!r}: bullets[{index}] is empty")
        if len(bullet) > max_chars:
            raise ValueError(
                f"layout={layout!r}: bullets[{index}] is {len(bullet)} characters "
                f"long, maximum is {max_chars}"
            )


class SlideBase(BaseModel):
    """Общая часть ЛЮБОЙ раскладки: заголовок, цитаты и правила про цитаты.

    Всё, что относится к цитатам (минимум одна, дедупликация, проверка на
    подмножество выданных чанков), живёт здесь и наследуется всеми пятью
    раскладками. Это не удобство, а единственный способ не потерять правило:
    скопированная в пять классов проверка держится ровно до появления шестой
    раскладки, которую напишут по образцу, забыв одну строку.

    extra="forbid" — тоже здесь, и он несёт двойную нагрузку. Первая очевидна:
    поле, которого нет в схеме, — это выдумка модели, и её незачем тащить
    дальше. Вторая важнее: именно запрет лишних полей превращает раскладки в
    ВЗАИМОИСКЛЮЧАЮЩИЕ. Без него слайд `{"layout": "metric", "bullets": [...]}`
    прошёл бы валидацию как metric, буллеты молча исчезли бы при рендере, и мы
    получили бы ровно то, ради чего всё затевалось наоборот: слайд, который
    модель задумала одним, а код нарисовал другим.
    """

    model_config = ConfigDict(extra="forbid")

    heading: _bounded_text(SLIDE_HEADING_MAX_CHARS)
    citations: list[SlideCitation] = Field(min_length=1)

    def digest_texts(self) -> list[str]:
        """Тексты слайда для дайджеста уже написанного (build_written_digest).

        Раскладки хранят текст в разных полях, и «взять slide.bullets» после
        этой волны означает потерять всё, что модель написала на слайдах других
        раскладок, — то есть вернуть повторы, ради борьбы с которыми дайджест и
        заведён (мера 2 правила «не добивать», см. prompts.py). Поэтому «что
        считать текстом слайда» знает сама раскладка, а не её потребитель.
        """
        raise NotImplementedError

    @model_validator(mode="after")
    def _deduplicate_citations(self, info: ValidationInfo) -> "SlideBase":
        """Схлопнуть повторы цитат внутри слайда, сохранив порядок первого.

        Модель охотно ссылается на один и тот же фрагмент из каждого буллета, и
        без этого шага дубли доезжали бы до слайда «Источники», до подсчёта
        использованных фрагментов и до любого будущего потребителя — каждому
        пришлось бы дедуплицировать у себя, и один из них однажды забыл бы.
        Поэтому чистка стоит в нормализаторе схемы: ниже по течению дублей
        просто не существует.

        Ключ — пара (source_id, chunk_id), а не один chunk_id: тот же chunk_id
        с чужим source_id — не дубль, а противоречие, и его обязана увидеть и
        отвергнуть проверка на подмножество ниже, а не проглотить дедупликация.
        """
        seen: set[tuple[int, str]] = set()
        unique: list[SlideCitation] = []
        for citation in self.citations:
            key = (citation.source_id, citation.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            unique.append(citation)
        self.citations = unique
        return self

    @model_validator(mode="after")
    def _check_citations_subset(self, info: ValidationInfo) -> "SlideBase":
        """Цитаты — только на чанки, реально переданные в промпт.

        allowed_citations приходит контекстом валидации: {chunk_id: source_id}
        ровно того набора, который сборщик промпта положил в сообщение. Ссылка
        мимо набора — это выдуманный источник, и слайд с ней недействителен
        целиком: частично «почистить» цитаты нельзя, потому что неизвестно,
        какое из утверждений слайда опиралось на выдуманный фрагмент.
        """
        allowed = (info.context or {}).get("allowed_citations")
        if allowed is None:
            return self
        for citation in self.citations:
            expected_source = allowed.get(citation.chunk_id)
            if expected_source is None:
                raise ValueError(
                    f"citation chunk_id={citation.chunk_id!r} was not provided in the "
                    f"context; allowed chunk_id values are "
                    f"{sorted(allowed)}"
                )
            if citation.source_id != expected_source:
                raise ValueError(
                    f"citation chunk_id={citation.chunk_id!r} belongs to "
                    f"source_id={expected_source}, not {citation.source_id}"
                )
        return self


class BulletsSlide(SlideBase):
    """Список независимых фактов — раскладка по умолчанию и самая частая.

    Ровно то, чем слайд был до введения раскладок, поэтому и границы у него
    прежние: менять их заодно значило бы смешать две правки в одной волне и
    потерять возможность сказать, от какой из них изменилась колода.
    """

    layout: Literal[LAYOUT_BULLETS]
    bullets: list[str] = Field(
        min_length=SLIDE_BULLETS_MIN, max_length=SLIDE_BULLETS_MAX
    )

    @model_validator(mode="after")
    def _check_bullets(self, info: ValidationInfo) -> "BulletsSlide":
        _check_bullet_list(
            self.bullets, layout=LAYOUT_BULLETS, max_chars=SLIDE_BULLET_MAX_CHARS
        )
        return self

    def digest_texts(self) -> list[str]:
        return list(self.bullets)


class CompareColumn(BaseModel):
    """Одна сторона сравнения: свой подзаголовок и свой короткий список."""

    model_config = ConfigDict(extra="forbid")

    heading: _bounded_text(SLIDE_COMPARE_HEADING_MAX_CHARS)
    bullets: list[str] = Field(
        min_length=SLIDE_COMPARE_BULLETS_MIN, max_length=SLIDE_COMPARE_BULLETS_MAX
    )

    @model_validator(mode="after")
    def _check_bullets(self, info: ValidationInfo) -> "CompareColumn":
        _check_bullet_list(
            self.bullets,
            layout=LAYOUT_COMPARE,
            max_chars=SLIDE_COMPARE_BULLET_MAX_CHARS,
        )
        return self


class CompareSlide(SlideBase):
    """Две стороны одного вопроса: было/стало, два режима, план/факт.

    Сторон ровно две и они именованные (left/right), а не список: сравнение
    трёх и более колонок — это таблица, а таблица на слайде набирается кеглем,
    которым её никто не прочитает. Список из двух элементов дал бы ту же форму,
    но потерял бы имена, а рендеру нужно знать, какая сторона слева.
    """

    layout: Literal[LAYOUT_COMPARE]
    left: CompareColumn
    right: CompareColumn

    def digest_texts(self) -> list[str]:
        # Подзаголовки колонок входят в дайджест наравне с их строками: «было /
        # стало» — это и есть содержание слайда, а не разметка.
        return [
            self.left.heading,
            *self.left.bullets,
            self.right.heading,
            *self.right.bullets,
        ]


class MetricSlide(SlideBase):
    """Одна величина, ради которой существует весь слайд.

    Величина остаётся СТРОКОЙ, а не числом, и это осознанно: «12,5 %»,
    «3,2 млрд сомонӣ», «с 1 января 2027» — в документах величина неотделима от
    единицы измерения и от способа её записи. Число плюс поле «единица» модель
    заполняла бы гаданием, а рендер собирал бы обратно, теряя запятую как
    десятичный разделитель и пробел как разделитель разрядов.
    """

    layout: Literal[LAYOUT_METRIC]
    value: _bounded_text(SLIDE_METRIC_VALUE_MAX_CHARS)
    caption: _bounded_text(SLIDE_METRIC_CAPTION_MAX_CHARS)
    # Уточнения может не быть — и это законный ответ, а не недоработка модели:
    # «12,5 %» под подписью часто исчерпывающи. Ключ разрешено не писать вовсе,
    # умолчание то же самое.
    note: _bounded_text(SLIDE_METRIC_NOTE_MAX_CHARS) | None = None

    @model_validator(mode="before")
    @classmethod
    def _blank_note_means_none(cls, data: Any) -> Any:
        """Пустая строка в note — это "уточнения нет", а не ошибка.

        В остальных полях пробельная строка отвергается (см. _reject_blank):
        там она означает нарисованную пустоту. Здесь пустоту рисовать не надо —
        поле необязательное, и "" с null совпадают по смыслу дословно. Отказ
        целого слайда из-за выбора между двумя способами сказать «ничего» —
        цена, за которую мы не получаем ничего.
        """
        if not isinstance(data, dict):
            return data
        note = data.get("note")
        if isinstance(note, str) and not note.strip():
            return {**data, "note": None}
        return data

    def digest_texts(self) -> list[str]:
        # Величина входит в дайджест вместе с подписью: без неё следующий слайд
        # не увидит, что эта цифра уже названа, и назовёт её второй раз.
        texts = [f"{self.value} — {self.caption}"]
        if self.note:
            texts.append(self.note)
        return texts


class SlideStep(BaseModel):
    """Один шаг процесса: чем он называется и что в нём происходит."""

    model_config = ConfigDict(extra="forbid")

    title: _bounded_text(SLIDE_STEP_TITLE_MAX_CHARS)
    text: _bounded_text(SLIDE_STEP_TEXT_MAX_CHARS)


class StepsSlide(SlideBase):
    """Упорядоченная последовательность: этапы, сроки, порядок действий.

    Порядок несёт смысл — им отличается процедура от списка, — поэтому шаги
    хранятся списком, а не словарём, и рендер обязан рисовать их в том порядке,
    в котором они пришли. Номера шагов модель не пишет: их ставит код, иначе
    «Шаг 2» из текста и вторая позиция в списке однажды разъедутся.
    """

    layout: Literal[LAYOUT_STEPS]
    steps: list[SlideStep] = Field(
        min_length=SLIDE_STEPS_MIN, max_length=SLIDE_STEPS_MAX
    )

    def digest_texts(self) -> list[str]:
        return [f"{step.title}: {step.text}" for step in self.steps]


class QuoteSlide(SlideBase):
    """Формулировка из документа, которая важна дословно.

    attribution — источник СЛОВАМИ («Налоговый кодекс, статья 12»), и это не
    дубликат цитат: citations остаются машинной ссылкой на фрагмент и рисуются
    слайдом «Источники», а attribution — это подпись под цитатой, которую
    читает человек. Имена файлов и служебные идентификаторы в неё не попадают
    (см. правило про идентификаторы в prompts.py).
    """

    layout: Literal[LAYOUT_QUOTE]
    text: _bounded_text(SLIDE_QUOTE_TEXT_MAX_CHARS)
    attribution: _bounded_text(SLIDE_QUOTE_ATTRIBUTION_MAX_CHARS)

    def digest_texts(self) -> list[str]:
        return [self.text]


# Слайд — это РАЗМЕЧЕННОЕ объединение по полю layout, а не одна схема со
# множеством необязательных полей.
#
# Разница не косметическая. Схема с необязательными полями принимает
# {"layout": "metric", "value": ..., "steps": [...]} и оставляет решение
# «что рисовать» рендереру — то есть тому, кто узнаёт о противоречии последним
# и молча. Размеченное объединение отвергает такой ответ на входе, а сообщение
# об отказе называет и раскладку, и поле ("metric.steps: Extra inputs are not
# permitted"), и потому годится для повторного промпта.
#
# Умолчания у layout нет НАМЕРЕННО. Молчаливый фолбэк на bullets вернул бы
# ровно ту колоду, ради которой всё это и затевалось наоборот — одинаковые
# слайды, — и скрыл бы главный симптом: модель не поняла, что от неё хотят.
# Отсутствие layout обязано быть видно как отказ.
PresentationSlide = Annotated[
    BulletsSlide | CompareSlide | MetricSlide | StepsSlide | QuoteSlide,
    Field(discriminator="layout"),
]

# Размеченное объединение — не класс, и model_validate у него нет. Разбор идёт
# через TypeAdapter; он собирается ОДИН раз на модуль, потому что сборка схемы
# pydantic не бесплатна, а слайд-вызовов в колоде до тринадцати.
SLIDE_ADAPTER = TypeAdapter(PresentationSlide)


def strip_code_fences(raw: str) -> str:
    """Снять ```-ограждения вокруг JSON.

    Берём тело ПЕРВОГО ограждённого блока, а не вырезаем все ``` из строки:
    модель нередко добавляет после блока пояснение, и склейка «до и после»
    превратила бы валидный JSON в мусор.
    """
    text = (raw or "").strip()
    match = _CODE_FENCE_RE.search(text)
    if match:
        return match.group("body").strip()
    return text


def escape_control_characters(text: str) -> str:
    """Привести неэкранированные управляющие символы ВНУТРИ строк к JSON-виду.

    Модель регулярно отдаёт JSON, в котором внутри строкового значения стоит
    живой перевод строки, — по стандарту это невалидный JSON, и `json.loads`
    отвергает его целиком («Invalid control character at ...»). Отказ честный,
    но лечится он не второй попыткой (на живом стенде оба захода упали на одном
    и том же символе), а здесь: до разбора.

    Что делаем внутри строки:
    * `\\n` и `\\t` — экранируем. Это ровно то, что модель имела в виду, и
      смысл текста сохраняется: перенос строки в буллете остаётся переносом.
    * остальные C0 — выкидываем. Смысла они не несут, а доехав до рендерера,
      становятся видимым мусором на слайде.

    Вне строк управляющие символы — законный пробельный разделитель JSON, и там
    текст не трогаем вообще. Отсюда же и главное свойство: валидный ответ
    проходит через функцию НЕИЗМЕННЫМ, потому что в валидном JSON живых
    управляющих символов внутри строк нет по определению.

    Альтернатива — `json.loads(..., strict=False)` — отвергнута сознательно:
    она не чистит, а разрешает, причём всему разбору сразу. Мусорный `\\x1b`
    молча доехал бы до текста слайда, а заодно ослабла бы проверка ответов, у
    которых с управляющими символами всё в порядке.
    """
    result: list[str] = []
    in_string = False
    after_backslash = False
    for char in text:
        if not in_string:
            if char == '"':
                in_string = True
            result.append(char)
            continue
        if after_backslash:
            # Символ под экранированием — уже часть escape-последовательности,
            # и своего значения (в том числе «закрыть строку») не имеет.
            result.append(char)
            after_backslash = False
            continue
        if char == "\\":
            result.append(char)
            after_backslash = True
            continue
        if char == '"':
            in_string = False
            result.append(char)
            continue
        if char < " ":
            replacement = _MEANINGFUL_CONTROL_ESCAPES.get(char)
            if replacement is not None:
                result.append(replacement)
            continue
        result.append(char)
    return "".join(result)


def parse_model_json(raw: str) -> dict[str, Any]:
    """Сырой ответ модели → словарь.

    Второй проход (от первой `{` до последней `}`) нужен из-за вступлений вида
    «Вот JSON:». Он намеренно ограничен объектом верхнего уровня: гадать по
    обрывкам, что модель имела в виду, — способ выдать испорченный ответ за
    рабочий.

    Чистка управляющих символов стоит здесь же, в одном месте с ограждениями и
    до `json.loads`, — чтобы оба прохода разбирали один и тот же текст.
    """
    text = escape_control_characters(strip_code_fences(raw))
    if not text:
        raise LlmResponseError("model returned an empty response")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise LlmResponseError(
                f"response is not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})"
            ) from exc
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError as inner_exc:
            raise LlmResponseError(
                f"response is not valid JSON: {inner_exc.msg} "
                f"(line {inner_exc.lineno}, column {inner_exc.colno})"
            ) from inner_exc

    if not isinstance(payload, dict):
        raise LlmResponseError(
            f"response must be a JSON object, got {type(payload).__name__}"
        )
    return payload


def format_validation_error(exc: ValidationError) -> str:
    """ValidationError → одна строка для повторного промпта.

    Текст обязан называть РАСКЛАДКУ и ПОЛЕ: он — единственная подсказка второй
    попытки, и «response does not match the required schema» отправляет модель
    гадать заново по всей схеме из пяти раскладок.

    С раскладками это выходит само: loc размеченного объединения начинается с
    тега, то есть «metric.value: String should have at most 24 characters» и
    «steps.steps.1.title: Field required» получаются без нашего участия.
    Отдельной обработки требует лишь одна ошибка — отсутствие самого layout:
    pydantic сообщает о ней с ПУСТЫМ loc («не смог извлечь тег»), и в общем
    виде она превратилась бы в «<root>: Unable to extract tag...», где не
    названо ни поле, ни допустимые значения.
    """
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error["loc"]) or "<root>"
        if error["type"] in ("union_tag_not_found", "union_tag_invalid"):
            discriminator = str((error.get("ctx") or {}).get("discriminator", "")).strip(
                "'"
            )
            # Обе ошибки про одно поле, и обе приходят с пустым loc: неизвестный
            # тег («layout: table») своё сообщение уже несёт, ему хватает
            # правильного имени места, а вот отсутствующий обязан ещё и
            # перечислить допустимые значения.
            if discriminator:
                location = discriminator
            # Список значений подставляется только для СВОЕГО дискриминатора:
            # появится в модуле второе размеченное объединение — оно получит
            # родное сообщение pydantic, а не чужие имена раскладок.
            if error["type"] == "union_tag_not_found" and discriminator == "layout":
                expected = ", ".join(repr(name) for name in SLIDE_LAYOUTS)
                parts.append(f"layout: field required, expected one of {expected}")
                continue
        # Та же услуга секции плана. Раскладка там — обычное поле, а не тег
        # объединения, и pydantic на её отсутствие говорит только «Field
        # required»: место названо (sections.2.layout), а чем его заполнить — не
        # сказано. Модели, забывшей поле целиком, перечень допустимых значений
        # нужен больше всех: сама она о нём и не вспомнила.
        if error["type"] == "missing" and error["loc"] and error["loc"][-1] == "layout":
            expected = ", ".join(repr(name) for name in SLIDE_LAYOUTS)
            parts.append(f"{location}: field required, expected one of {expected}")
            continue
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts) or "response does not match the required schema"


def validate_plan(raw: str, *, slide_count: int) -> PresentationPlan:
    """Ответ модели → план ровно на (slide_count - 2) контентных секций."""
    payload = parse_model_json(raw)
    try:
        return PresentationPlan.model_validate(
            payload,
            context={"expected_sections": content_section_count(slide_count)},
        )
    except ValidationError as exc:
        raise LlmResponseError(format_validation_error(exc)) from exc


def validate_slide(raw: str, *, allowed_citations: dict[str, int]) -> SlideBase:
    """Ответ модели → слайд ОДНОЙ из раскладок, с цитатами внутри allowed_citations.

    allowed_citations — {chunk_id: source_id} тех чанков, что ушли в промпт.
    Какая раскладка вернулась, потребитель узнаёт по полю layout; общее у всех —
    heading, citations и digest_texts().
    """
    payload = parse_model_json(raw)
    try:
        return SLIDE_ADAPTER.validate_python(
            payload,
            context={"allowed_citations": allowed_citations},
        )
    except ValidationError as exc:
        raise LlmResponseError(format_validation_error(exc)) from exc
