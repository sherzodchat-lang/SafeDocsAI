import asyncio
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from app.core.exceptions import ExternalServiceError, SettingsError, SettingsErrors
from app.shared.settings.config import settings

logger = logging.getLogger(__name__)


# Вид модели. Списки моделей чата и эмбеддингов раньше были одним и тем же
# списком «всё, что установлено в Ollama», и валидация проверяла только факт
# установки — то есть не защищала ни от чего: в поле модели чата проходил
# qwen3-embedding:8b (чат ломается на каждом запросе), а в поле эмбеддингов —
# gemma4:26b (имя коллекции ChromaDB выводится из embedding-модели, поиск
# молча уезжает в пустую коллекцию).
#
# Чем отличать. Ollama в /api/tags — это ровно то, что читает
# ModelManager.list_ollama_models, — отдаёт по каждой модели имя, digest,
# размер и details: family, families, parameter_size, quantization_level.
# Признака «модель считает эмбеддинги» там нет ни в каком виде: у
# qwen3-embedding:8b семейство такое же, как у чат-моделей qwen, а размерность
# вектора не отдаётся вовсе. Единственный надёжный признак у Ollama —
# capabilities (["embedding"] против ["completion"]) из /api/show, но это
# отдельный запрос НА КАЖДУЮ установленную модель, а каталог собирается на
# каждом открытии админ-панели и на каждом сохранении настроек.
#
# Поэтому классификация идёт по имени и намеренно НЕ бинарная: запрещаем
# только то, что опознано уверенно, а неопознанное остаётся разрешённым в
# обоих полях. Иначе модель с нестандартным именем (своя сборка, приватный
# реестр, ollama create) стала бы невыбираемой, и админ не смог бы настроить
# систему вообще — ошибка «слишком строго» здесь дороже пропущенной опечатки.
MODEL_KIND_CHAT = "chat"
MODEL_KIND_EMBEDDING = "embedding"
MODEL_KIND_UNKNOWN = "unknown"

# Подстрока в любом месте имени: nomic-embed-text, qwen3-embedding:8b,
# granite-embedding, snowflake-arctic-embed, mxbai-embed-large.
_EMBEDDING_SUBSTRINGS = ("embed",)

# Отдельным словом (имя режется по не-буквенно-цифровым): bge-m3,
# multilingual-e5-large, all-minilm, LaBSE, gte-qwen2, paraphrase-multilingual.
# Именно словом, а не префиксом: "e5" префиксом поймало бы "e5b" из тега
# размера (gemma4:e4b и родня), и чат-модель уехала бы в embedding-список.
_EMBEDDING_WORDS = frozenset(
    {"bge", "gte", "e5", "minilm", "labse", "sbert", "sentence", "paraphrase"}
)

# Префикс ПЕРВОГО слова имени: gemma4:26b, llama3.1:8b, qwen2.5-coder,
# phi4-mini, gpt-oss:20b. Только первого — у embedding-моделей имя семейства
# часто стоит вторым словом (gte-qwen2), и по любому слову они попали бы в
# чат-список раньше, чем сработал бы маркер эмбеддинга.
_CHAT_NAME_PREFIXES = (
    "aya",
    "codellama",
    "codegemma",
    "codeqwen",
    "codestral",
    "command",
    "deepseek",
    "devstral",
    "dolphin",
    "exaone",
    "falcon",
    "gemma",
    "glm",
    "gpt",
    "granite",
    "hermes",
    "internlm",
    "llama",
    "llava",
    "magistral",
    "minicpm",
    "mistral",
    "mixtral",
    "moondream",
    "nemotron",
    "olmo",
    "openchat",
    "orca",
    "phi",
    "qwen",
    "smollm",
    "solar",
    "stablelm",
    "starcoder",
    "tinyllama",
    "tulu",
    "vicuna",
    "wizardlm",
    "yi",
    "zephyr",
)

_MODEL_WORD_RE = re.compile(r"[^a-z0-9]+")

# Границы окна контекста (num_ctx) для chat_model_num_ctx и
# contextual_embedding_num_ctx.
#
# Верхняя граница была 262144 — предел архитектуры, а не стенда, и он ничего не
# защищал. num_ctx задаёт размер KV-кэша, а тот занимает видеопамять СВЕРХ
# весов: 262144 на gemma4:26b (18 ГБ весов) раздувает кэш на порядок, модель
# выталкивается в CPU, генерация упирается в 120-секундный таймаут
# (OLLAMA_TIMEOUT_SECONDS), фолбэк идёт с тем же окном, и пользователь получает
# 502 — то есть значение из формы ломало работу, а форма его принимала молча.
#
# 32768 — с запасом над всем, что реально развёрнуто, и на порядок ниже
# прежнего предела:
#   * Modelfile'ы стенда пиннят num_ctx 20000 (gemma4:e4b) и 12000
#     (gemma4:26b, gemma4:31b) — см. DEPLOY.md, волна 4;
#   * умолчания в коде: 20000 для чата, 8192 для контекстного обогащения,
#     12288 у ModelManager, когда num_ctx не передали;
#   * бюджет промпта в app/api/deps.py посчитан от 12000-20000 токенов.
# То есть 32768 не мешает поднять окно относительно любого сегодняшнего
# значения (запас 1.6x к самому большому), но не даёт задать величину, которой
# на этом железе физически нет места.
#
# Нижняя граница прежняя: меньше 2048 не хватает даже на системную часть
# промпта, и модель отвечает обрывками.
MIN_NUM_CTX = 2048
MAX_NUM_CTX = 32768

# Слова, которые считаются логическим значением. Общие у снисходительного
# чтения и у строгой записи: разойдись они — файл, записанный через API, мог бы
# перестать читаться так же, как записан.
_TRUE_WORDS = frozenset({"1", "true", "yes", "on"})
_FALSE_WORDS = frozenset({"0", "false", "no", "off"})


class RuntimeSettingsService:
    DEFAULTS: dict[str, Any] = {
        "chat_model": settings.OLLAMA_MODEL_CHAT,
        "embedding_model": settings.OLLAMA_MODEL_EMBEDDING,
        "retrieval_top_k": 20,
        "top_k": 5,
        "default_domain_profile": "tax",
        "enable_condense_query": True,
        "contextual_embedding_enabled": False,
        # Пусто — «модель не выбрана», а не «возьмите вот эту».
        #
        # Здесь стояло "gemma3:4b" — модель, которой нет ни в одном
        # развёртывании (на стенде gemma4:e4b, gemma4:26b, qwen3-embedding:8b).
        # Поле при этом валидировалось по каталогу ВСЕГДА, поэтому любое
        # сохранение упиралось в модель выключенной функции: поправить top_k
        # было нельзя, пока не переставишь значение в визуально пустом селекте.
        # Умолчание чат-модели сюда тоже не годится: contextual_embedding_model
        # — самостоятельный выбор (обогащение гоняет модель на КАЖДЫЙ чанк, и
        # берут туда модель поменьше), а подставленное молча значение выглядело
        # бы как сделанный админом выбор.
        #
        # Пустое значение безопасно: индексация читает
        # `if _ctx_enabled and _ctx_model` (app/modules/documents/service.py),
        # то есть без модели обогащение просто не выполняется. Чтобы
        # переключатель не оказался включённым впустую, включение с пустой
        # моделью отвергается (SettingsErrors.CONTEXTUAL_MODEL_REQUIRED).
        "contextual_embedding_model": "",
        "chat_model_num_ctx": 20000,
        "contextual_embedding_num_ctx": 8192,
        "reranker_enabled": False,
        "reranker_model": "gemma4:e4b",
        # Признак «векторы посчитаны прежней embedding-моделью». Ставится при
        # смене embedding_model и снимается только полностью успешной
        # переиндексацией (POST /api/v1/documents/reindex).
        "reindex_required": False,
    }

    # Блокировка read-modify-write.
    #
    # update_settings читает файл целиком и целиком же его переписывает. Два
    # сохранения, попавшие в одно окно, затирают друг друга: побеждает то, что
    # записалось вторым, а правка первого пропадает без следа.
    #
    # asyncio.Lock, а не threading.Lock: воркер один (backend/run.py, по
    # умолчанию --workers 1), эндпоинты асинхронные, и конкуренция здесь
    # ровно та, что бывает внутри одного цикла событий. При нескольких
    # воркерах uvicorn этого недостаточно — см. комментарий к _write_settings.
    #
    # Лок заводится лениво и привязан к своему циклу: asyncio.Lock с версии
    # 3.10 запоминает цикл на первом ожидании и на чужом падает
    # («is bound to a different event loop»). В приложении цикл один, а в
    # тестах IsolatedAsyncioTestCase поднимает новый на каждый тест — там
    # каждый получит свой лок, и это верно: чужой цикл уже не работает.
    _write_lock: asyncio.Lock | None = None
    _write_lock_loop: Any = None

    @classmethod
    def _lock(cls) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if cls._write_lock is None or cls._write_lock_loop is not loop:
            cls._write_lock = asyncio.Lock()
            cls._write_lock_loop = loop
        return cls._write_lock

    @classmethod
    def _settings_path(cls) -> Path:
        backend_dir = Path(__file__).resolve().parents[3]
        path = backend_dir / "data" / "runtime_settings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def available_models(cls) -> list[str]:
        return cls.model_catalog()["available_models"]

    @classmethod
    def classify_model(cls, model: str) -> str:
        """Вид модели по её имени: chat, embedding или unknown.

        Порядок проверок важен: маркер эмбеддинга сильнее семейства. Иначе
        gte-qwen2 и qwen3-embedding опознавались бы как чат-модели qwen.
        """
        normalized = str(model or "").strip().lower()
        if not normalized:
            return MODEL_KIND_UNKNOWN
        if any(marker in normalized for marker in _EMBEDDING_SUBSTRINGS):
            return MODEL_KIND_EMBEDDING
        words = [word for word in _MODEL_WORD_RE.split(normalized) if word]
        if not words:
            return MODEL_KIND_UNKNOWN
        if _EMBEDDING_WORDS.intersection(words):
            return MODEL_KIND_EMBEDDING
        if words[0].startswith(_CHAT_NAME_PREFIXES):
            return MODEL_KIND_CHAT
        return MODEL_KIND_UNKNOWN

    @classmethod
    def model_catalog(cls) -> dict[str, Any]:
        ollama_available = True
        ollama_error: str | None = None
        candidates: list[str] = []

        try:
            candidates.extend(
                __import__("app.modules.rag.model_manager", fromlist=["ModelManager"])
                .ModelManager()
                .list_ollama_models()
            )
        except ExternalServiceError as exc:
            ollama_available = False
            ollama_error = exc.message
        except Exception as exc:  # noqa: BLE001 - см. ниже
            # Ловилось только ExternalServiceError, а операций в блоке три, и
            # своё исключение бросает лишь третья:
            #   * динамический импорт тянет пакет ollama (ImportError);
            #   * конструктор ModelManager создаёт двух ollama.Client по адресу
            #     из OLLAMA_API_BASE и на мусорном адресе падает сам;
            #   * и только list_ollama_models заворачивает свои отказы.
            # Итог: опечатка в OLLAMA_API_BASE давала 500 на
            # GET /api/v1/settings/ — то есть гасила ровно тот экран, на
            # котором эту опечатку и правят. Каталог обязан собираться всегда,
            # пусть и пустым: без него не открыть настройки, а без настроек не
            # починить каталог.
            ollama_available = False
            ollama_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Model catalog is unavailable (%s: %s); settings will open with "
                "empty model lists",
                type(exc).__name__,
                exc,
                exc_info=True,
            )

        installed = cls._unique_models(candidates)
        kinds = {model: cls.classify_model(model) for model in installed}
        # Неопознанная модель (MODEL_KIND_UNKNOWN) попадает в ОБА списка: см.
        # комментарий к MODEL_KIND_* выше — надёжного признака у Ollama нет,
        # и запрет по эвристике оставил бы админа без выбора.
        available_chat_models = [
            model for model in installed if kinds[model] != MODEL_KIND_EMBEDDING
        ]
        available_embedding_models = [
            model for model in installed if kinds[model] != MODEL_KIND_CHAT
        ]
        return {
            "available_models": installed,
            "available_chat_models": available_chat_models,
            "available_embedding_models": available_embedding_models,
            "ollama_available": ollama_available,
            "ollama_error": ollama_error,
        }

    @classmethod
    def _validate_model_choice(
        cls, model: str, kind: str, label: str, catalog: dict[str, Any]
    ) -> None:
        """Проверить, что модель годится для поля своего вида.

        Разные коды и тексты для «не установлена» и «установлена, но не того
        вида»: первое лечится `ollama pull`, второе — выбором другой модели, и
        клиенту нужно показать разные подсказки.
        """
        # «Каталога нет» — это не «модели нет». Признак ollama_available
        # считался, но при записи не использовался: лежащая Ollama отдавала
        # пустые списки, и сохранение УЖЕ настроенной и стоящей на месте модели
        # отвергалось как «модель не установлена» — админ шёл её доставлять,
        # хотя чинить надо было сервис. Отказ здесь другой по смыслу: тело
        # запроса верное, повторить его надо как есть (503, см.
        # _ERROR_STATUS в app/api/endpoints/settings.py).
        if not catalog.get("ollama_available", True):
            reason = catalog.get("ollama_error") or "Ollama did not answer"
            raise SettingsError(
                SettingsErrors.MODEL_CATALOG_UNAVAILABLE,
                f"Cannot verify {label} {model}: the list of installed models is "
                f"unavailable ({reason}). Repeat the request once Ollama is back.",
            )
        if kind == MODEL_KIND_EMBEDDING:
            allowed = catalog["available_embedding_models"]
            opposite = catalog["available_chat_models"]
            wrong_kind = f"{model} is a chat model and cannot be used for embeddings"
        else:
            allowed = catalog["available_chat_models"]
            opposite = catalog["available_embedding_models"]
            wrong_kind = f"{model} is an embedding model and cannot be used for chat"
        if model in allowed:
            return
        if model in opposite:
            raise SettingsError(SettingsErrors.MODEL_WRONG_KIND, wrong_kind)
        raise SettingsError(
            SettingsErrors.MODEL_NOT_INSTALLED, f"Unsupported {label}: {model}"
        )

    @classmethod
    def get_settings(cls) -> dict[str, Any]:
        """Настройки, дополненные умолчаниями и приведённые к нужным типам.

        Это ПУТЬ ЧТЕНИЯ, и он снисходителен намеренно: значение, ставшее
        негодным (профиль убрали из реестра, модель удалили из Ollama, в файле
        осталось num_ctx от прежних границ), чинится на лету — подменяется
        умолчанием или подрезается в диапазон, — и чтение продолжается. Падать
        здесь нельзя ни на чём: get_settings зовут чат, поиск, индексация и сам
        экран настроек, то есть отказ на испорченном значении погасил бы в том
        числе тот экран, на котором его исправляют.

        Строгость живёт в ПУТИ ЗАПИСИ (update_settings): там негодное значение
        отвергается с машинным кодом, а не подменяется молча. Разведены они по
        функциям: `_normalize_*` — чтение, `_require_*` — запись.

        Отсутствующий или испорченный файл — не отдельная ветка с коротким
        `return dict(DEFAULTS)`: тот возвращал словарь БЕЗ ключа "model", и
        обработчик GET /api/v1/settings/ падал на нём с KeyError, то есть
        админ-панель не открывалась ни разу до первого сохранения настроек.
        Пустой словарь вместо содержимого файла проходит тот же путь, и набор
        ключей на выходе один и тот же при любом состоянии диска.
        """
        path = cls._settings_path()
        data: Any = {}
        if path.exists():
            # Молчаливого `except Exception: data = {}` здесь больше нет.
            # Откат на умолчания — это ДРУГАЯ embedding-модель, то есть другая
            # коллекция ChromaDB: поиск уезжает в пустоту, а в журнале не
            # остаётся ни строки, по которой это можно связать с файлом
            # настроек. Отсутствие файла — норма (чистая установка, сюда мы
            # вообще не заходим), нечитаемый файл — авария.
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                # Файл исчез между exists() и чтением: сброс настроек прошёл
                # прямо сейчас. Не авария — ровно то же состояние, что и
                # чистая установка.
                logger.info(
                    "Runtime settings file %s disappeared while reading; "
                    "falling back to defaults",
                    path,
                )
            except (OSError, ValueError) as exc:
                # ValueError накрывает json.JSONDecodeError и UnicodeDecodeError.
                logger.error(
                    "Runtime settings file %s is unreadable (%s: %s). Falling "
                    "back to DEFAULTS — including embedding_model=%s, то есть "
                    "другую коллекцию ChromaDB: поиск будет отвечать пустотой, "
                    "пока файл не починят.",
                    path,
                    type(exc).__name__,
                    exc,
                    cls.DEFAULTS["embedding_model"],
                )

        if data and not isinstance(data, dict):
            logger.error(
                "Runtime settings file %s holds %s instead of an object; "
                "falling back to DEFAULTS",
                path,
                type(data).__name__,
            )

        merged = dict(cls.DEFAULTS)
        merged.update(data if isinstance(data, dict) else {})
        legacy_model = str(merged.get("model") or "").strip()
        merged["chat_model"] = str(merged.get("chat_model") or legacy_model).strip()
        merged["embedding_model"] = str(merged.get("embedding_model") or "").strip()
        merged["retrieval_top_k"] = cls._normalize_retrieval_top_k(
            merged.get("retrieval_top_k")
        )
        merged["top_k"] = cls._normalize_top_k(merged.get("top_k"))
        merged["enable_condense_query"] = cls._normalize_bool(
            merged.get("enable_condense_query"), default=True, field="enable_condense_query"
        )
        merged["contextual_embedding_enabled"] = cls._normalize_bool(
            merged.get("contextual_embedding_enabled"),
            default=False,
            field="contextual_embedding_enabled",
        )
        merged["contextual_embedding_model"] = str(
            merged.get("contextual_embedding_model") or ""
        ).strip()
        merged["chat_model_num_ctx"] = cls._normalize_num_ctx(
            merged.get("chat_model_num_ctx"), cls.DEFAULTS["chat_model_num_ctx"], "chat_model_num_ctx"
        )
        merged["contextual_embedding_num_ctx"] = cls._normalize_num_ctx(
            merged.get("contextual_embedding_num_ctx"),
            cls.DEFAULTS["contextual_embedding_num_ctx"],
            "contextual_embedding_num_ctx",
        )
        merged["reranker_enabled"] = cls._normalize_bool(
            merged.get("reranker_enabled"), default=False, field="reranker_enabled"
        )
        merged["reranker_model"] = str(merged.get("reranker_model") or cls.DEFAULTS["reranker_model"]).strip()
        merged["reindex_required"] = cls._normalize_bool(
            merged.get("reindex_required"), default=False, field="reindex_required"
        )
        if not merged["chat_model"]:
            merged["chat_model"] = cls.DEFAULTS["chat_model"]
        if not merged["embedding_model"]:
            merged["embedding_model"] = cls.DEFAULTS["embedding_model"]
        # "model" — устаревшее имя chat_model: под ним ключ лежит в старых
        # runtime_settings.json и его же читают чат и ask
        # (`.get("chat_model") or .get("model")`). Всегда выводим из
        # chat_model, а не храним в DEFAULTS отдельной строкой: два
        # независимых умолчания одного и того же значения разъедутся, и
        # админ-панель показывала бы модель, с которой никто не работает.
        merged["model"] = merged["chat_model"]
        merged["default_domain_profile"] = cls._normalize_domain_profile(
            merged.get("default_domain_profile")
        )
        return merged

    @classmethod
    def update_settings(cls, patch: dict[str, Any]) -> dict[str, Any]:
        """Применить патч к настройкам и сохранить их.

        Смена embedding_model требует подтверждения (`confirm_reindex`) — см.
        _require_reindex_confirmation. Проверка стоит ЗДЕСЬ, а не в обработчике
        HTTP: тогда мимо неё не пройдёт ни один путь, включая тот, который
        заведут завтра.

        Ключ confirm_reindex — признак операции, а не настройка: в файл он не
        попадает (`current` собирается из get_settings и явных присваиваний).

        Это ПУТЬ ЗАПИСИ: значение, которого система не понимает, отвергается с
        машинным кодом. Тихая подмена здесь была хуже отказа — клиент получал
        200 OK и полное тело настроек, а сохранено оказывалось другое (см.
        _normalize_domain_profile: "banking" превращался в "tax", то есть молча
        менялись правила ответов ассистента). Снисходительность осталась ровно
        на чтении — см. get_settings.
        """
        current = cls.get_settings()

        # Каталог собирается запросом в Ollama: берём его не более одного раза
        # за сохранение и только если он кому-то понадобился. Лениво, а не по
        # списку ключей в патче: нужен ли каталог для contextual_embedding_model,
        # заранее не известно — это зависит от ИТОГОВОГО состояния
        # переключателя обогащения, а не от состава патча.
        catalog_cache: dict[str, Any] = {}

        def catalog() -> dict[str, Any]:
            if not catalog_cache:
                catalog_cache.update(cls.model_catalog())
            return catalog_cache

        if "chat_model" in patch or "model" in patch:
            selected_model = str(
                patch.get("chat_model") or patch.get("model") or ""
            ).strip()
            if not selected_model:
                raise SettingsError(
                    SettingsErrors.MODEL_REQUIRED, "Chat model must not be empty"
                )
            cls._validate_model_choice(
                selected_model, MODEL_KIND_CHAT, "chat model", catalog()
            )
            current["chat_model"] = selected_model
            current["model"] = selected_model
        if "embedding_model" in patch:
            embedding_model = str(patch["embedding_model"] or "").strip()
            if not embedding_model:
                raise SettingsError(
                    SettingsErrors.MODEL_REQUIRED, "Embedding model must not be empty"
                )
            cls._validate_model_choice(
                embedding_model, MODEL_KIND_EMBEDDING, "embedding model", catalog()
            )
            old_embedding = current.get("embedding_model", "")
            if old_embedding and old_embedding != embedding_model:
                # Порядок важен: сначала имя модели проверено по каталогу, и
                # только потом спрашивается подтверждение. Иначе на опечатку в
                # имени клиент получал бы «подтвердите переиндексацию» вместо
                # «такой модели нет».
                cls._require_reindex_confirmation(
                    patch, old_embedding, embedding_model
                )
                current["reindex_required"] = True
                logger.warning(
                    "Embedding model changed from %s to %s — reindex required. "
                    "Run POST /api/v1/documents/reindex to rebuild the vector store.",
                    old_embedding,
                    embedding_model,
                )
            current["embedding_model"] = embedding_model
        if "retrieval_top_k" in patch:
            current["retrieval_top_k"] = cls._require_int_in_range(
                patch["retrieval_top_k"], "retrieval_top_k", 1, 50
            )
        if "top_k" in patch:
            current["top_k"] = cls._require_int_in_range(patch["top_k"], "top_k", 1, 20)
        if "default_domain_profile" in patch:
            current["default_domain_profile"] = cls._require_domain_profile(
                patch["default_domain_profile"]
            )
        if "enable_condense_query" in patch:
            current["enable_condense_query"] = cls._require_bool(
                patch["enable_condense_query"], "enable_condense_query"
            )

        # --- Контекстное обогащение ---
        #
        # Модель проверяется по ИТОГОВОМУ состоянию, а не по составу патча.
        # Проверка стояла безусловно, и это ломало сохранение целиком: поле
        # выключенной функции держало модель, которой в Ollama нет, и 400
        # приходил на любой патч, вплоть до правки одного top_k.
        #
        # Считать «включено» по патчу тоже нельзя — переключатель может лежать
        # в сохранённых настройках, а в патче быть только модель (и наоборот).
        # Отсюда два вопроса подряд: включено ли обогащение ПОСЛЕ патча и
        # делает ли этот патч выбор моделью живым.
        contextual_enabled_before = bool(current["contextual_embedding_enabled"])
        contextual_model_before = str(current["contextual_embedding_model"] or "")
        if "contextual_embedding_enabled" in patch:
            current["contextual_embedding_enabled"] = cls._require_bool(
                patch["contextual_embedding_enabled"], "contextual_embedding_enabled"
            )
        if "contextual_embedding_model" in patch:
            current["contextual_embedding_model"] = str(
                patch["contextual_embedding_model"] or ""
            ).strip()
        contextual_enabled = bool(current["contextual_embedding_enabled"])
        contextual_model = str(current["contextual_embedding_model"] or "")
        # Проверяем ровно два случая: модель меняют при включённом обогащении и
        # обогащение включают этим патчем. Перепроверять уже сохранённую модель
        # на каждом сохранении нельзя — тогда правка top_k при включённом
        # обогащении снова зависела бы от Ollama и от того, не удалили ли
        # модель из неё, то есть вернулась бы та же неисправность.
        if contextual_enabled and (
            contextual_model != contextual_model_before or not contextual_enabled_before
        ):
            if not contextual_model:
                raise SettingsError(
                    SettingsErrors.CONTEXTUAL_MODEL_REQUIRED,
                    "Contextual embedding is enabled but no model is selected: "
                    "indexing would silently skip enrichment. Pick a chat model "
                    "or turn contextual embedding off.",
                )
            # Вопреки имени поля это ЧАТ-модель: она пишет текстовое описание
            # чанка перед эмбеддингом (см. _generate_llm_context), а вектор
            # считает embedding_model. Поэтому и проверяется по списку чата.
            cls._validate_model_choice(
                contextual_model,
                MODEL_KIND_CHAT,
                "contextual embedding model",
                catalog(),
            )

        if "chat_model_num_ctx" in patch:
            current["chat_model_num_ctx"] = cls._require_num_ctx(
                patch["chat_model_num_ctx"], "chat_model_num_ctx"
            )
        if "contextual_embedding_num_ctx" in patch:
            current["contextual_embedding_num_ctx"] = cls._require_num_ctx(
                patch["contextual_embedding_num_ctx"], "contextual_embedding_num_ctx"
            )
        if "reranker_enabled" in patch:
            current["reranker_enabled"] = cls._require_bool(
                patch["reranker_enabled"], "reranker_enabled"
            )
        if "reranker_model" in patch:
            model = str(patch["reranker_model"] or "").strip()
            current["reranker_model"] = model or cls.DEFAULTS["reranker_model"]

        cls._write_settings(current)
        return current

    # --- Подтверждение смены embedding-модели ---------------------------

    @classmethod
    def _require_reindex_confirmation(
        cls, patch: dict[str, Any], old_model: str, new_model: str
    ) -> None:
        """Не дать сменить embedding-модель мимоходом.

        Имя коллекции ChromaDB выводится из embedding-модели, поэтому смена —
        не «настройка», а операция над всем поиском: сразу после сохранения
        запросы уходят в коллекцию, которую никто не заполнял, и система
        отвечает так, будто документов нет вовсе. Вернуть их может только
        полная переиндексация.

        Поэтому подтверждение спрашивается на сервере, а не рисуется галочкой
        на клиенте: клиентских путей к PUT /api/v1/settings/ несколько
        (форма, автосохранение, чужой скрипт), и любой из них не должен уметь
        задеть эту настройку заодно с соседними.
        """
        # Строго, как и всё остальное в патче: confirm_reindex — согласие на
        # операцию, и «включить» его мусорным значением (bool("banana") — это
        # True) нельзя.
        if "confirm_reindex" in patch and cls._require_bool(
            patch["confirm_reindex"], "confirm_reindex"
        ):
            return
        raise SettingsError(
            SettingsErrors.REINDEX_CONFIRMATION_REQUIRED,
            f"Changing the embedding model ({old_model} -> {new_model}) moves "
            f"search to another ChromaDB collection and requires a full "
            f"reindex. Repeat the request with confirm_reindex=true.",
        )

    # --- Операции, меняющие файл настроек -------------------------------
    #
    # Асинхронные обёртки: read-modify-write целиком идёт под _lock(), поэтому
    # два запроса не затирают правки друг друга. Синхронный update_settings
    # оставлен как есть — им пользуются скрипты вне API (backend/*.py), где
    # цикла событий нет.

    @classmethod
    async def update_settings_locked(cls, patch: dict[str, Any]) -> dict[str, Any]:
        async with cls._lock():
            return cls.update_settings(patch)

    @classmethod
    async def reset_settings(cls, *, confirm_reindex: bool = False) -> dict[str, Any]:
        """Вернуть все настройки к умолчаниям.

        Сброс возвращает к умолчанию и embedding_model, то есть имеет ровно те
        же последствия, что и её смена вручную, — и требует того же
        подтверждения. Спрашивается оно только когда модель и правда меняется:
        сброс, ничего не меняющий в embedding_model (а это обычный случай —
        её никто не трогал), подтверждения не требует, иначе клиент приучится
        слать confirm_reindex=true всегда.
        """
        async with cls._lock():
            current = cls.get_settings()
            old_embedding = current.get("embedding_model", "")
            new_embedding = cls.DEFAULTS["embedding_model"]
            embedding_changes = bool(old_embedding) and old_embedding != new_embedding
            if embedding_changes and not confirm_reindex:
                raise SettingsError(
                    SettingsErrors.REINDEX_CONFIRMATION_REQUIRED,
                    f"Resetting settings returns the embedding model to "
                    f"{new_embedding} ({old_embedding} -> {new_embedding}) and "
                    f"requires a full reindex. Repeat the request with "
                    f"confirm_reindex=true.",
                )

            restored = dict(cls.DEFAULTS)
            # Флаг не сбрасывается заодно с остальным: он описывает состояние
            # ChromaDB, а не настройку. Долг за прежней сменой модели остаётся
            # долгом, и сброс добавляет к нему свой, если увёл модель обратно.
            restored["reindex_required"] = (
                bool(current.get("reindex_required")) or embedding_changes
            )
            cls._write_settings(restored)
            logger.info(
                "Runtime settings reset to defaults (embedding_model %s -> %s, "
                "reindex_required=%s)",
                old_embedding,
                new_embedding,
                restored["reindex_required"],
            )
            return cls.get_settings()

    @classmethod
    async def clear_reindex_required(cls) -> bool:
        """Снять флаг после успешной переиндексации; вернуть новое значение.

        Файл не переписывается, если флага и не было: на чистой установке
        переиндексация иначе создавала бы runtime_settings.json на ровном
        месте — то есть замораживала бы текущие умолчания в файле.
        """
        async with cls._lock():
            current = cls.get_settings()
            if not current.get("reindex_required"):
                return False
            current["reindex_required"] = False
            cls._write_settings(current)
            logger.info(
                "Reindex completed successfully — reindex_required cleared "
                "(embedding_model=%s)",
                current.get("embedding_model"),
            )
            return False

    # --- Запись ---------------------------------------------------------

    @classmethod
    def _write_settings(cls, data: dict[str, Any]) -> None:
        """Сохранить настройки целиком, не показывая читателю полуфабрикат.

        Было `path.write_text(...)`: он открывает файл на запись, то есть
        СНАЧАЛА обрезает его в ноль, и только потом наполняет. Читатель,
        попавший в это окно (а читают настройки все — чат, поиск,
        индексация), получал обрезанный JSON, молча откатывался на умолчания
        и уезжал в другую коллекцию ChromaDB.

        Теперь содержимое пишется во временный файл и переставляется на место
        через os.replace: на POSIX это атомарный rename — читатель видит либо
        прежний файл целиком, либо новый целиком, третьего состояния нет.
        Временный файл создаётся в ТОМ ЖЕ каталоге: rename атомарен только в
        пределах одной файловой системы.

        Атомарность записи не заменяет блокировку (_lock) и тем более не
        решает вопрос нескольких воркеров uvicorn: os.replace не даст увидеть
        полуфайл, но два процесса, прочитавших настройки одновременно,
        по-прежнему затрут правки друг друга — для этого нужна блокировка,
        общая для процессов (файловый flock на этом же файле или перенос
        настроек в БД).
        """
        path = cls._settings_path()
        payload = json.dumps(data, ensure_ascii=False, indent=2)
        descriptor, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f"{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                # Без fsync содержимое может остаться в кеше страниц: после
                # внезапной перезагрузки на месте окажется файл нужной длины
                # из нулей. Настройки пишутся редко, цена незаметна.
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            # Иначе каждая неудачная запись оставляла бы в data/ мусор
            # runtime_settings.json.*.tmp.
            try:
                os.unlink(tmp_name)
            except OSError:  # pragma: no cover - уже удалён или недоступен
                pass
            raise

    # --- Чтение: снисходительные нормализаторы ---------------------------
    #
    # Вызываются ТОЛЬКО из get_settings. Их задача — отдать рабочее значение из
    # того, что лежит в файле, чем бы оно ни оказалось: файл могли править
    # руками, он мог остаться от прежней версии (num_ctx за нынешними
    # границами), а профиль или модель могли исчезнуть уже после сохранения.
    # Отказ здесь погасил бы всё сразу, включая экран настроек.
    #
    # Подменять молча им всё же не положено: о каждой подмене остаётся строка в
    # журнале — иначе «настройка не работает» снова нечем объяснить.

    @classmethod
    def _normalize_top_k(cls, value: Any) -> int:
        return cls._clamp_on_read(value, cls.DEFAULTS["top_k"], 1, 20, "top_k")

    @classmethod
    def _normalize_retrieval_top_k(cls, value: Any) -> int:
        return cls._clamp_on_read(
            value, cls.DEFAULTS["retrieval_top_k"], 1, 50, "retrieval_top_k"
        )

    @classmethod
    def _normalize_num_ctx(cls, value: Any, default: int, field: str = "num_ctx") -> int:
        return cls._clamp_on_read(value, default, MIN_NUM_CTX, MAX_NUM_CTX, field)

    @classmethod
    def _clamp_on_read(
        cls, value: Any, default: int, minimum: int, maximum: int, field: str
    ) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            if value is not None:
                cls._report_read_fallback(field, value, default)
            return default
        clamped = max(minimum, min(number, maximum))
        if clamped != number:
            cls._report_read_fallback(field, number, clamped)
        return clamped

    @classmethod
    def _normalize_domain_profile(cls, value: Any) -> str:
        profile = str(value or "").strip().lower()
        if profile in cls._domain_profiles():
            return profile
        fallback = cls.DEFAULTS["default_domain_profile"]
        if profile:
            # Профиль был, но в реестре его больше нет: его убрали уже после
            # сохранения настроек. Правила ответов ассистента при этом
            # меняются, и молчать об этом нельзя — но и падать нельзя тоже.
            cls._report_read_fallback("default_domain_profile", profile, fallback)
        return fallback

    @classmethod
    def _normalize_bool(cls, value: Any, default: bool, field: str = "flag") -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in _TRUE_WORDS:
                return True
            if normalized in _FALSE_WORDS:
                return False
        if value is None:
            return default
        # Ни истина, ни ложь. Что имел в виду написавший — неизвестно, поэтому
        # читаем как получится (bool), но след оставляем: через API такое
        # значение больше не проходит, а в файле оно откуда-то взялось.
        coerced = bool(value)
        cls._report_read_fallback(field, value, coerced)
        return coerced

    # Об одном и том же значении жалуемся один раз за жизнь процесса:
    # get_settings зовут на каждый вопрос к ассистенту и на каждый чанк при
    # индексации (app/modules/documents/service.py), и без этого фильтра одна
    # строка в файле залила бы журнал.
    _reported_read_fallbacks: set[str] = set()

    @classmethod
    def _report_read_fallback(cls, field: str, value: Any, fallback: Any) -> None:
        signature = f"{field}={value!r}"
        if signature in cls._reported_read_fallbacks:
            return
        cls._reported_read_fallbacks.add(signature)
        logger.warning(
            "Runtime settings: %s=%r is not a usable value; reading it as %r. "
            "Saving this value through the API is refused — fix it in the admin "
            "panel or in %s.",
            field,
            value,
            fallback,
            cls._settings_path(),
        )

    # --- Запись: строгие проверки ----------------------------------------
    #
    # Вызываются ТОЛЬКО из update_settings. Здесь значение не «лежит в файле», а
    # прислано прямо сейчас, и подменить его молча значит ответить 200 OK на
    # выбор, которого админ не делал: он видит успех, а работает система
    # по-другому.

    @staticmethod
    def _require_int_in_range(value: Any, field: str, minimum: int, maximum: int) -> int:
        if isinstance(value, float) and not value.is_integer():
            raise SettingsError(
                SettingsErrors.INVALID_NUMBER,
                f"{field} must be a whole number, got {value!r}",
            )
        try:
            number = int(value)
        except (TypeError, ValueError):
            raise SettingsError(
                SettingsErrors.INVALID_NUMBER,
                f"{field} must be a whole number, got {value!r}",
            ) from None
        if not minimum <= number <= maximum:
            raise SettingsError(
                SettingsErrors.VALUE_OUT_OF_RANGE,
                f"{field} must be between {minimum} and {maximum}, got {number}",
            )
        return number

    @classmethod
    def _require_num_ctx(cls, value: Any, field: str) -> int:
        """Окно контекста: отказ вместо клампа.

        Кламп здесь был худшим из вариантов: 0 превращался в 2048, а 262144 —
        в само себя, и в обоих случаях приходило 200 OK. Первое молча урезает
        окно втрое против выбранного, второе раздувает KV-кэш до размера, под
        который на этом железе нет памяти (см. MIN_NUM_CTX/MAX_NUM_CTX).
        """
        return cls._require_int_in_range(value, field, MIN_NUM_CTX, MAX_NUM_CTX)

    @classmethod
    def _require_domain_profile(cls, value: Any) -> str:
        """Профиль домена: отказ вместо тихой подмены.

        Здесь стояло `return profile if profile in available else "tax"`, и
        PUT {"default_domain_profile": "banking"} отвечал 200 с профилем "tax":
        админ видел успех, а правила ответов ассистента менялись на другие.
        """
        profile = str(value or "").strip().lower()
        available = cls._domain_profiles()
        if profile not in available:
            raise SettingsError(
                SettingsErrors.UNSUPPORTED_DOMAIN_PROFILE,
                f"Unsupported domain profile: {value!r}. Available: "
                f"{', '.join(sorted(available))}",
            )
        return profile

    @staticmethod
    def _require_bool(value: Any, field: str) -> bool:
        """Логическое поле: отказ вместо bool(что угодно).

        `bool("banana")` — это True, то есть переключатель включался бы от
        любого мусора, и в ответе admin увидел бы честное true, которого не
        просил.
        """
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in _TRUE_WORDS:
                return True
            if normalized in _FALSE_WORDS:
                return False
        raise SettingsError(
            SettingsErrors.INVALID_BOOLEAN,
            f"{field} must be a boolean, got {value!r}",
        )

    @staticmethod
    def _domain_profiles() -> set[str]:
        from app.domain_profiles import list_domain_profiles as _list_profiles

        return set(_list_profiles())

    @staticmethod
    def _unique_models(candidates: list[str]) -> list[str]:
        seen: set[str] = set()
        unique_models: list[str] = []
        for model in candidates:
            normalized = str(model or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique_models.append(normalized)
        return unique_models
