# 🏛️ AndozAI

Локальный ИИ-помощник по налогам.

## 📋 Технологический стек

| Компонент | Технология |
|-----------|------------|
| Frontend | React 19 + Tailwind CSS + Vite |
| Backend | FastAPI (Python 3.12) |
| Векторная БД | ChromaDB |
| База данных | PostgreSQL + asyncpg |
| Локальная ИИ | Gemma 3n (Ollama) |
| OCR | Tesseract |

## 🚀 Быстрый старт

### Предварительные требования

- Python 3.12+
- Node.js 20+
- PostgreSQL 14+
- Ollama (с моделью gemma3n:e4b)
- Tesseract OCR

### 1. Клонирование и настройка

```bash
cd soliqai

# Backend
cd backend
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# или: venv\Scripts\activate  # Windows

pip install -r requirements.txt

# Настройка переменных окружения
cp .env.example .env
# Отредактируйте .env файл
```

### 2. Настройка базы данных

```bash
# Создать базу данных PostgreSQL
createdb andozai_db

# Инициализация таблиц
cd backend
source venv/bin/activate
python -c "import asyncio; from app.core.database import init_db; asyncio.run(init_db())"
```

### 3. Запуск backend

```bash
cd backend
source venv/bin/activate

# Режим разработки
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001

# Или для production
uvicorn app.main:app --host 0.0.0.0 --port 8001 --workers 4
```

API будет доступен по адресу: http://localhost:8001

Документация API: http://localhost:8001/api/v1/docs

### 4. Запуск frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend будет доступен по адресу: http://localhost:5173

### 5. Настройка Ollama

```bash
# Установка Ollama (если не установлен)
curl -fsSL https://ollama.com/install.sh | sh

# Загрузка модели
ollama pull gemma3n:e4b

# Запуск сервера
ollama serve
```

## 🔒 Безопасность

### Перед production-развертыванием:

1. **Измените SECRET_KEY** в `.env`:
   ```bash
   openssl rand -hex 32
   ```

2. **Ограничьте CORS** в `.env`:
   ```
   CORS_ORIGINS=http://localhost:5173,https://yourdomain.com
   ```

3. **Используйте сильные пароли** для базы данных

4. **Установите переменную окружения**:
   ```
   ENVIRONMENT=production
   ```

## 🧪 Тестирование

```bash
cd backend
source venv/bin/activate
python -m unittest discover -v tests/
```

## 📊 Health Checks

- `GET /health` - Базовая проверка работоспособности
- `GET /ready` - Проверка готовности всех зависимостей (БД, ChromaDB)

## 📁 Структура проекта

```
soliqai/
├── backend/
│   ├── app/
│   │   ├── api/           # API endpoints
│   │   ├── core/          # Конфигурация, БД, безопасность
│   │   ├── models/        # SQLModel модели
│   │   └── services/      # Бизнес-логика
│   ├── tests/             # Юнит-тесты
│   ├── .env.example       # Шаблон переменных окружения
│   └── requirements.txt   # Python зависимости
├── frontend/
│   ├── src/
│   │   ├── pages/         # React страницы
│   │   ├── components/    # UI компоненты
│   │   └── services/      # API клиент
│   └── package.json
└── README.md
```

## 🔧 Переменные окружения

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `OLLAMA_API_BASE` | Базовый URL Ollama API | `http://localhost:11434` |
| `OLLAMA_MODEL_CHAT` | Модель Ollama для чата | `gemma3n:e4b` |
| `OLLAMA_MODEL_EMBEDDING` | Модель Ollama для эмбеддингов | **нет умолчания**, см. ниже |
| `SECRET_KEY` | JWT секретный ключ | Автогенерация |
| `POSTGRES_*` | Настройки PostgreSQL | localhost |
| `CORS_ORIGINS` | Разрешенные origins | * |
| `ENVIRONMENT` | Среда (development/production) | development |

### Модель эмбеддингов — обязательная настройка

`OLLAMA_MODEL_EMBEDDING` умолчания не имеет, и это сделано намеренно. Имя
коллекции ChromaDB выводится из названия модели, а ChromaDB на незнакомое имя
коллекции не отказывает, а создаёт пустую. Умолчание в коде означало бы, что
процесс, поднятый без этой переменной, молча уходит в чужую пустую коллекцию:
поиск ничего не находит при полной базе, и в логах при этом пусто. Так на
стенде и завелась лишняя пустая коллекция `andozai_docs_nomic_embed_text`.

Порядок разрешения: `backend/data/runtime_settings.json` (то, что выбрано в
админ-панели) → переменная `OLLAMA_MODEL_EMBEDDING` → отказ.

Отказ мягкий: бэкенд стартует, `GET/PUT /api/v1/settings/` работают (иначе
модель негде выбрать), а всё, чему нужна векторная база — чат, поиск,
индексация, удаление векторов, — отвечает `503` с кодом
`settings.embedding_model_unset`. Очередь индексации при этом не рушится:
задачи остаются в очереди, документы — в `pending`, и очередь трогается сама,
как только модель выбрана.

При старте бэкенд пишет в журнал одну строку о том, чем считаются векторы:

```
embedding_model=qwen3-embedding:8b -> коллекция andozai_docs_qwen3_embedding_8b, векторов 12480
embedding_model не задан -> коллекции нет, векторов нет. Задайте модель ...
```

Задавайте ту модель, которой векторы уже посчитаны: её смена требует полной
переиндексации (`POST /api/v1/documents/reindex`) и подтверждается отдельно.

## 👥 Роли пользователей

- **admin** - Полный доступ (документы, FAQ, настройки, пользователи)
- **content_manager** - Управление документами и FAQ
- **user** - Только чат

## 📝 Лицензия
