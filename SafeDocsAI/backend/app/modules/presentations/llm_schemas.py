"""Схемы ответов модели для генерации презентаций и их разбор.

Этап 0 (прототип пайплайна). Здесь живут три вещи, которые дальше переедут в
сервис без изменений: форма плана, форма слайда и разбор сырого ответа модели.

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
SLIDE_BULLETS_MIN = 3
SLIDE_BULLETS_MAX = 5

_CODE_FENCE_RE = re.compile(
    r"```[ \t]*[a-zA-Z0-9_+-]*[ \t]*\r?\n(?P<body>.*?)(?:\r?\n)?[ \t]*```",
    re.DOTALL,
)


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
    source_id: int
    chunk_id: str

    @model_validator(mode="before")
    @classmethod
    def _coerce_chunk_id(cls, data: Any) -> Any:
        # chunk_id в системе — строка (идентификатором вектора в ChromaDB служит
        # str(chunk.id)), но модель почти всегда отдаёт его числом, потому что
        # видит в промпте цифры. Приводим сами: это разночтение формата, а не
        # выдуманная ссылка, и отвергать из-за него целый слайд нечестно —
        # проверка на подмножество ниже от этого не слабеет.
        if isinstance(data, dict) and isinstance(data.get("chunk_id"), int):
            data = {**data, "chunk_id": str(data["chunk_id"])}
        return data


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


def parse_model_json(raw: str) -> dict[str, Any]:
    """Сырой ответ модели → словарь.

    Второй проход (от первой `{` до последней `}`) нужен из-за вступлений вида
    «Вот JSON:». Он намеренно ограничен объектом верхнего уровня: гадать по
    обрывкам, что модель имела в виду, — способ выдать испорченный ответ за
    рабочий.
    """
    text = strip_code_fences(raw)
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
