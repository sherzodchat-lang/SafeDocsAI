# Ускорение retrieval в semantic search pipeline на FastAPI + PostgreSQL + ChromaDB

## Исходная архитектура и ключевые источники задержек

По описанию, текущий retrieval-пайплайн выглядит так:

**query normalization → query condensation (Ollama) → FAQ matching → Chroma query → distance filtering**, плюс **второй retrieval для таджикского языка с RU hint**.

Критически важно: в асинхронном FastAPI/Starlette любые **синхронные** вызовы (HTTP-клиент, SDK, вызовы Chroma-клиента, CPU-тяжёлая постобработка) внутри `async def` могут **блокировать event loop**, что резко ухудшает tail latency под нагрузкой (P95/P99). Starlette/FastAPI в ряде случаев уводит синхронный код в thread pool, но если вы напрямую вызываете блокирующие функции из `async def`, блокировка остаётся вашей ответственностью. citeturn0search0turn3search0turn3search10

Также важно понимать особенности Chroma:

- В Chroma по умолчанию используется ANN-индекс **HNSW**, а параметры (например, `ef_search`, `num_threads`, метрика `space`) существенно влияют на **скорость/качество** поиска. citeturn5view2turn1search9  
- У Chroma есть настройка **какие поля включать в ответ** (`include`), и это прямо влияет на объём данных/сериализацию/пересылку. citeturn1search6turn1search15

По Ollama:

- Ollama REST API по `/api/generate` (и ряду других) стримит ответы по умолчанию; для короткого structured-output/condense зачастую выгоднее отключать streaming (`"stream": false`), чтобы упростить обработку и снизить накладные расходы на потоковую сборку. citeturn0search2  
- В официальной библиотеке есть `AsyncClient` на базе `httpx.AsyncClient`, чтобы **не блокировать event loop** на HTTP-вызовах. citeturn2search0

Ограничение анализа: у меня нет доступа к вашему репозиторию/фрагментам файлов по путям `soliqai/...`, поэтому ниже — **максимально практичный план и patch-level шаблоны** “куда и что вставить” по указанным файлам и диапазонам строк, но без точного совпадения с вашими именами функций/переменных.

## Оценочная декомпозиция latency по этапам

Ниже — **оценочные** цифры (ориентиры), чтобы прикинуть, где ROI максимален. Реальные значения зависят от: модели в Ollama, железа (CPU/GPU), размера коллекции Chroma, режима Chroma (embedded vs http server), размера `top_k`, и того, где считается embedding (внутри Chroma через `embedding_function` или снаружи).

### Базовая картина (типичная для local LLM + vector DB)

| Этап | Что происходит | P50 (оценка) | P95 (оценка) | Почему P95 “раздувается” |
|---|---|---:|---:|---|
| Query normalization | нормализация текста, возможно regex/юникод | 1–5 ms | 3–15 ms | редко проблема; может стать CPU-bound при агрессивных regex |
| Query condensation (Ollama) | LLM вызов, короткий вывод (condensed query) | 80–250 ms | 250–900 ms | очередь в Ollama, холодный KV-cache, отсутствие batching, CPU инференс |
| FAQ matching | точное/нечёткое совпадение или отдельный mini-retrieval | 5–25 ms | 20–80 ms | походы в БД, отсутствие индексов/кеша |
| Chroma query | ANN поиск top_k | 15–80 ms | 40–220 ms | disk I/O, высокий `ef_search`, большой payload `include`, конкуренция потоков |
| Post-filter (distance threshold, merge, sort) | фильтрация по distance<=1.8, дедуп | 1–12 ms | 5–35 ms | если много кандидатов/сложная логика |

### Дополнительная цена таджикского “второго retrieval с RU hint”

Если второй retrieval запускается **всегда**, добавляете примерно:

- + (Chroma query + post-filter) = **+20–120 ms P50**, **+60–300 ms P95**  
- плюс потенциально + время на генерацию/подготовку RU hint (если это тоже LLM/правила)

Ключевой вывод по ROI: чаще всего самые большие “куски” — **condense** и **второй retrieval**, а под нагрузкой — ещё и **event loop blocking**, который резко раздувает P95/P99. citeturn0search0turn3search0turn2search0

## План ускорения, отсортированный по ROI

Пояснение к процентам: это **ожидаемое** снижение именно retrieval latency (без генерации финального ответа), относительно вашего текущего поведения. “Качество” — влияние на recall/precision retrieval (не на качество генерации).  

### Сводная таблица способов

| ROI | Уровень | Способ | Ожидаемое снижение latency (P50 / P95) | Влияние на качество retrieval | Сложность | Риски |
|---:|---|---|---|---|---|---|
| 1 | Quick win | **Гейтинг condense**: пропускать condensation для коротких/простых запросов + авто-fallback | **−20–45% / −25–55%** | Обычно нейтрально или лучше (меньше “переформулировок”), при неверном гейте может ухудшить recall | S | Нужны эвристики, иначе редкие деградации на “сложных” запросах |
| 2 | Quick win | **Кэш condense** (TTL+LRU) по `(normalized_query, lang, user_segment)` | **−10–35% / −15–45%** | Нейтрально | S | Нужна защита от взрыва ключей, учитывать персонализацию |
| 3 | Quick win | **Условный таджикский RU-hint retrieval**: запускать 2-й поиск только если 1-й дал “плохой” сигнал | **−10–30% / −15–40%** (в среднем по трафику) | Нейтрально/слегка хуже на части кейсов, но можно сохранить качество через умный триггер | M | Неправильный триггер → пропуск полезного RU fallback |
| 4 | Quick win | **Один Chroma query вместо двух**: отправлять **две query_embeddings** за один запрос и потом мерджить | **−5–20% / −8–25%** | Нейтрально | M | Нужно аккуратно слияние и дедуп; зависит от того, как сейчас устроен RU hint |
| 5 | Quick win | **Сделать Ollama вызов неблокирующим**: `AsyncClient` + timeouts + reuse клиента | **−0–10% / −10–35%** (под нагрузкой сильнее) | Нейтрально | S | Неправильные таймауты → рост ошибок/фолбэков; нужно корректно обрабатывать отмену |
| 6 | Quick win | **Убрать повторную инициализацию Chroma client/collection** (singleton на lifespan) | **−5–25% / −10–30%** | Нейтрально | S–M | Ошибки жизненного цикла (shutdown), многопроцессный деплой (uvicorn workers) |
| 7 | Quick win | **Не блокировать event loop на Chroma query**: `anyio.to_thread.run_sync()`/`run_in_threadpool()` для sync клиента | **P50: ≈0–5%**, но **P95: −15–50%** под concurrency | Нейтрально | M | Threadpool starvation (по умолчанию ~40 токенов), нужен контроль лимитов citeturn0search0turn3search0 |
| 8 | Quick win | **Сократить payload Chroma**: `include` только нужных полей, не тащить embeddings, по возможности не тащить documents | **−5–20% / −5–25%** | Нейтрально, если документы подтягиваются отдельно; иначе может потребоваться доп. fetch | S–M | Если уйти от `documents`, нужен быстрый lookup (Postgres по ids батчем) citeturn1search6turn1search15 |
| 9 | Medium | **Параллелизация** независимых частей: FAQ matching и (часть) подготовки query/embeddings | **−5–20% / −10–25%** | Нейтрально | M | Сложнее отладка, нужно аккуратно с таймаутами/отменой |
| 10 | Medium | **Тюнинг HNSW `ef_search`, `num_threads`** под ваши latency цели | **−5–30% / −10–40%** | `ef_search` вниз → возможное падение recall; `num_threads` вверх → обычно нейтрально | M | Нужно измерять; неправильно — потеря качества или рост CPU citeturn5view2turn1search9 |
| 11 | Medium | **Сегментация коллекций / where-фильтры**: меньше поисковое пространство (tenant, domain, doc_type) | **−5–25% / −5–30%** | Часто лучше precision, иногда лучше скорость | M–L | Нужно аккуратно поддерживать метаданные/фильтры; риск “не туда отфильтровали” |
| 12 | Medium | **Дефрагментация/repair HNSW индекса** (если много update/delete) | **−0–20% / −5–30%** | Может восстановить точность и скорость | M | Операция обслуживания, нужно окно/процедуры citeturn3search2 |
| 13 | Architecture | **Перейти на мультиязычный embedding** и убрать RU-hint как класс | **−10–30% / −10–40%** | Обычно лучше на cross-lingual | L | Требуется переэмбеддинг коллекции и пересмотр метрик/порогов |
| 14 | Architecture | **Смена vector store на более “прод” решение** (Qdrant/Milvus/pgvector и т.д.) с промышленным async-клиентом/индексами | **−10–50% / −20–60%** (в tail часто сильнее) | Может улучшиться, если правильно настроить | L | Миграция данных, эксплуатация, риски совместимости |
| 15 | Architecture | **Вынести retrieval в отдельный сервис** (изоляция CPU/IO, отдельный пул потоков/процессов, warm caches) | **P95: −20–60%** при нагрузке | Нейтрально | L | Операционная сложность, сетевые hop-ы, трассировка |

Ключевые причинно-следственные связи, на которых держится план:
- **Event loop нельзя блокировать**: для блокирующих операций используйте worker threads (`anyio.to_thread.run_sync`, `run_in_threadpool`) или нативные async-клиенты. По умолчанию размер thread pool ограничен (AnyIO limiter часто 40), и это нужно учитывать при переносе синхронных вызовов в threads. citeturn0search0turn3search0turn3search10  
- В Chroma **HNSW тюнинг** — главный рычаг компромисса “скорость ↔ recall”, и параметры (`space`, `ef_search`, `num_threads`) явно задокументированы. citeturn5view2turn1search9  
- Для Chroma query есть управляемые `include` поля, что позволяет уменьшать объём ответа и накладные расходы. citeturn1search6turn1search15  
- Для Ollama есть **`AsyncClient`**, а streaming можно отключить, что упрощает и иногда ускоряет короткие ответы. citeturn2search0turn0search2

## Patch-level изменения для топовых ускорений

Ниже — **конкретные правки уровня “что менять”** в указанных файлах/диапазонах. Поскольку исходников у меня нет, это шаблоны, которые вы адаптируете под ваши имена функций.

### Гейтинг condensation и быстрый fallback

**Где**: `soliqai/backend/app/api/endpoints/chat.py:177-317` (основной пайплайн).  
**Идея**: вызывать condense только когда это реально помогает.

Пример эвристики (дешёвая, но эффективная):
- если запрос ≤ N токенов/слов, без явных анафор (“он/она/это/там/здесь/выше”), без сложных инструкций — **не конденсировать**;
- если язык `tj` и/или запрос содержит много шумовых символов — condense допускается.

Псевдо-патч:
- добавьте функцию `should_condense(normalized_query, lang, has_history)`  
- если `False` → `condensed_query = normalized_query`  
- если `True` → вызывайте Ollama в timeout.

Практический эффект: вы убираете десятки/сотни миллисекунд на большинстве коротких запросов.

Опора на источники: используйте async-вызов Ollama (ниже) и корректные таймауты, чтобы не блокировать event loop. citeturn2search0turn3search10

### Кэш condensation (TTL+LRU) и дедупликация “в полёте”

**Где**: логичнее вынести в `soliqai/backend/app/services/rag_service.py` рядом с функцией condensation (по вашим диапазонам `52-77` или рядом с тем местом, где вызывается Ollama).  
**Что сделать**:
- Ввести in-memory cache (например, `cachetools.TTLCache(maxsize=10_000, ttl=3600)`), ключ: `(lang, normalized_query)` или `(lang, normalized_query, tenant)` если есть.
- Добавить “singleflight”: если 20 запросов пришли одновременно с одним ключом — выполняем один вызов Ollama, остальные ждут future.

Важно: кэш должен жить **на процесс** (uvicorn worker). Если у вас много воркеров, кэш будет per-worker — это всё равно даёт выигрыш.

### Условный второй retrieval для таджикского языка (качественный триггер)

**Где**: `chat.py:177-317` (после первого Chroma retrieval + filter) и, вероятно, логика в `rag_service.py:246-297`.  
**Что изменить**:
- Сейчас (по описанию) второй retrieval с RU hint делается “для таджикского языка”. Сделайте его **условным**:

Триггеры, которые почти всегда коррелируют с “первый retrieval плохой”:
- `len(results_after_filter) < k_min` (например, < 2)  
- `min_distance > threshold_bad` (если L2/косинус понятны), или `avg_distance` слишком высокий  
- “плоское распределение” distance (нет явного лидера)

Тогда:
- если retrieval хороший → **не** делать RU fallback  
- если плохой → делать RU fallback (или делать его параллельно, но “заканчивать рано”, если первый дал хороший сигнал)

Это обычно сохраняет качество, но убирает постоянную прибавку к latency.  

### Переиспользование клиентов: Ollama AsyncClient и Chroma client/collection как singletons

**Где**:
- Ollama: место создания клиента в `chat.py:94-116` или `chat.py:150-157` (судя по вашему списку — там могут быть зависимости/инициализации).
- Chroma: скорее всего в `rag_service.py` и/или где создаётся `chromadb.Client()/PersistentClient()/HttpClient()`.

**Что сделать**:
- Создать клиентов один раз на старте приложения (FastAPI lifespan/startup) и положить в `app.state`.
- Для Ollama использовать `AsyncClient` (официальный), чтобы не блокировать event loop на HTTP. citeturn2search0  
- Для Chroma: не создавать `PersistentClient`/`HttpClient` на каждый запрос — это и лишние накладные расходы, и потенциально плохие эффекты по памяти/ресурсам. В экосистеме Chroma встречаются кейсы, когда создание `PersistentClient` “per request” приводит к проблемам с освобождением памяти. citeturn1search10turn1search5

Шаблон на FastAPI lifespan (идея):
- в `app = FastAPI(lifespan=...)` создать:
  - `app.state.ollama = AsyncClient(host=..., timeout=...)`
  - `app.state.chroma_client = chromadb.PersistentClient(path=...)` или `HttpClient(...)`
  - `app.state.collection = client.get_collection(...)`

### Не блокировать event loop на Chroma query + контроль thread pool

Если Chroma-клиент синхронный (что часто бывает), то внутри `async def`:
- оборачивайте `collection.query(...)` в `await anyio.to_thread.run_sync(...)`  
- следите за лимитом thread pool (по умолчанию AnyIO limiter часто 40); иначе под нагрузкой можете получить очередь задач и рост tail latency. citeturn0search0turn3search0turn3search10

При необходимости увеличить лимит — делайте это осознанно на старте (и только если у вас реально много блокирующих задач, и хватает CPU/RAM). citeturn3search0turn0search0

### Сократите `include` в Chroma query и уменьшите сериализацию/объём ответа

**Где**: `rag_service.py` вокруг вызова `collection.query`.  
**Что сделать**:
- Убедиться, что вы **не запрашиваете embeddings** в ответе.
- Если ваш downstream может работать по `ids/metadatas` (а документы подтягивать батчем из Postgres) — можно ещё сильнее “облегчить” Chroma response.

Chroma API поддерживает `include` (например `["documents","metadatas","distances"]` или более узко). citeturn1search6turn1search15

Это часто даёт простой выигрыш (и CPU, и сеть, и JSON-парсинг).

## Benchmark-план для доказуемого ускорения

Цель: сравнивать **до/после** и не потерять retrieval quality.

### Метрики latency и системные метрики

Рекомендуемый минимум:

1) **End-to-end retrieval latency** (без генерации ответа):
- `retrieval_total_seconds` (Histogram)

2) **Stage latencies** (Histogram на каждый этап):
- `normalize_seconds`
- `condense_seconds`
- `faq_match_seconds`
- `chroma_query_seconds`
- `post_filter_seconds`
- `tj_ru_fallback_seconds` (если есть)
- `chroma_payload_bytes` (Summary/Histogram) — полезно для эффекта `include`

3) **Качество и “поведение системы”**:
- `retrieval_results_before_filter` / `after_filter`
- `fallback_rate_tj_to_ru` (доля запросов, где включился RU hint)
- `cache_hit_rate_condense` (hits/misses)
- `top1_distance`, `topk_min_distance`, `avg_distance` (можно sampled)

4) **Async/конкурентность**:
- **event loop lag** (например, периодический таск, меряющий задержку scheduler-а)
- `threadpool_queueing` косвенно: время ожидания `to_thread` (если оборачивать и мерить)

Про перцентили в Prometheus/Grafana: используйте `histogram_quantile(0.5/0.95/0.99, rate(..._bucket[5m]))`. citeturn2search4

### Набор запросов для честного сравнения

Соберите фиксированный “evaluation pack”:

- **Не менее 300–1000 запросов** (лучше из логов прод/стейдж).
- Стратификация:
  - RU: короткие (1–5 слов), средние, длинные
  - TJ: короткие/средние/длинные
  - “Плохие” запросы: шумные, с опечатками, с сокращениями
  - Запросы с контекстом (если condensation зависит от истории)
- Для 100–200 запросов создать **golden labels**: набор релевантных doc-ids (или chunk-ids) вручную/полуавтоматически.

### Как сравнивать “до/после”

1) **Offline прогон** (без нагрузки):
- прогнать весь pack 3–5 раз (1-й прогревочный)
- сравнивать P50/P95/P99 по total и по этапам

2) **Load test** (важно для event loop проблем):
- фиксировать concurrency (например 10/50/100) и RPS
- сравнивать tail latency и процент timeouts
- отдельно смотреть “threadpool starvation” симптоматику (рост очередей/ступенчатые задержки) — это типично, если блокирующий код уехал в threads, но лимит остался дефолтным. citeturn0search0turn3search0

3) **Quality regression**:
- Recall@k, MRR@k по labeled subset
- `fallback_rate` (если вы делаете условный RU hint)
- распределение `top1_distance` (чтобы случайно не “сломать” шкалу distance сменой метрики/эмбеддингов)

## Как не блокировать event loop и как убрать повторную инициализацию Chroma-клиента

### Как избежать блокировки event loop

Практический чеклист:

- Любой вызов, который:
  - делает синхронный HTTP (`requests`, sync SDK),
  - делает синхронный доступ к диску/БД,
  - выполняет тяжёлый CPU кусок,
  
  **нельзя** выполнять напрямую в `async def`.

Решения:

1) Перейти на нативный async-клиент, где возможно:
- Для Ollama: `AsyncClient` (официальный), он использует `httpx.AsyncClient`. citeturn2search0  

2) Для синхронных вызовов — вынос в worker thread:
- `await anyio.to_thread.run_sync(blocking_func, ...)` (в Starlette/FastAPI это стандартный путь). citeturn0search0turn3search10  

3) Контролировать лимит потоков:
- AnyIO default limiter часто 40; при большом числе одновременных блокирующих задач будет очередь и рост tail latency. При необходимости лимит можно увеличить на старте. citeturn0search0turn3search0  

### Как убрать повторную инициализацию Chroma-клиента

Рекомендация уровня production:

- Создавайте Chroma client и collection **один раз** на старте приложения (FastAPI lifespan/startup).
- Храните их в `app.state` или в singleton-сервисе, который создаётся на startup и внедряется через зависимости.

Почему:
- `PersistentClient` предназначен для работы с локальным persist-dir; его логика/объекты тяжёлые, и создание per-request — лишние накладные расходы. citeturn1search10  
- В реальном мире встречаются баг-репорты, где pattern “создать PersistentClient на запрос” приводит к проблемам с памятью/освобождением ресурсов. Даже если это не ваш кейс, это сильный сигнал, что лучше держать клиент как долгоживущий объект. citeturn1search5  

Дополнительно:
- Если вы используете PostgreSQL через asyncpg и вдруг где-то создаёте pool/коннект per-request — это тоже надо убрать: pool должен создаваться на старте и переиспользоваться. citeturn2search13  

Если вы пришлёте (вставкой в чат) фрагменты кода из указанных диапазонов строк, я смогу:
- разметить конкретные `await`/`to_thread` точки,
- точно предложить, где параллелить FAQ/condense,
- и выдать “почти готовые” диффы под вашу кодовую базу.