"""Схемы ответов модели для генерации презентаций и их разбор.

Здесь живут три вещи: форма плана, форма слайда и разбор сырого ответа модели.
Модуль заведён на этапе 0 (прототип) и на этапе 1 переехал в пайплайн без
переписывания — правки касаются только границ и нормализации.

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
from typing import Any

from pydantic import (
    BaseModel,
    Field,
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
    heading: str = Field(min_length=1, max_length=SECTION_HEADING_MAX_CHARS)
    search_query: str = Field(min_length=1, max_length=SECTION_SEARCH_QUERY_MAX_CHARS)


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


class PresentationSlide(BaseModel):
    heading: str = Field(min_length=1, max_length=SLIDE_HEADING_MAX_CHARS)
    bullets: list[str] = Field(
        min_length=SLIDE_BULLETS_MIN, max_length=SLIDE_BULLETS_MAX
    )
    citations: list[SlideCitation] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_bullets(self, info: ValidationInfo) -> "PresentationSlide":
        for index, bullet in enumerate(self.bullets):
            if not bullet.strip():
                raise ValueError(f"bullets[{index}] is empty")
            if len(bullet) > SLIDE_BULLET_MAX_CHARS:
                raise ValueError(
                    f"bullets[{index}] is {len(bullet)} characters long, "
                    f"maximum is {SLIDE_BULLET_MAX_CHARS}"
                )
        return self

    @model_validator(mode="after")
    def _deduplicate_citations(self, info: ValidationInfo) -> "PresentationSlide":
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
    def _check_citations_subset(self, info: ValidationInfo) -> "PresentationSlide":
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
    """ValidationError → одна строка для повторного промпта."""
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error["loc"]) or "<root>"
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


def validate_slide(raw: str, *, allowed_citations: dict[str, int]) -> PresentationSlide:
    """Ответ модели → слайд, цитаты которого лежат внутри allowed_citations.

    allowed_citations — {chunk_id: source_id} тех чанков, что ушли в промпт.
    """
    payload = parse_model_json(raw)
    try:
        return PresentationSlide.model_validate(
            payload,
            context={"allowed_citations": allowed_citations},
        )
    except ValidationError as exc:
        raise LlmResponseError(format_validation_error(exc)) from exc
