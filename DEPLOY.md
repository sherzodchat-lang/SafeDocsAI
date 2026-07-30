---
name: thunder-compute-safedocsai-aigov
description: "Деплой SafeDocsAI на Thunder Compute — параллельные волны, ~20 мин с моделями"
metadata: 
  node_type: memory
  type: project
  originSessionId: b960a28a-edbd-4391-983d-d5609c426b43
  modified: 2026-07-30T16:05:00.000Z
---

# Деплой SafeDocsAI на Thunder Compute

> Прогнано 2026-07-29 и 2026-07-30 на A100-80GB / 64 GB RAM / Ubuntu + Python 3.12 + PostgreSQL 14.

Инстанс каждый раз новый и пустой — это всегда деплой с нуля. Внутри волны команды идут параллельно, следующая волна — после предыдущей.

**Сначала определить, где вы находитесь.** Агент может быть запущен как на локальной машине, так и прямо внутри инстанса:

```bash
hostname   # instance-<uuid>-main → вы ВНУТРИ инстанса
```

Изнутри инстанса **весь префикс `ssh tnr-0 "…"` убирается**, команды выполняются напрямую; правило про `setsid nohup` всё равно остаётся в силе для демонов. Волна 0 при этом сводится к пробросу порта (см. ниже), а `git clone` в волне 1 обычно уже не нужен — репозиторий лежит в `~/Aigov`.

**Три правила:**
1. Фон только так: `ssh -n tnr-0 "setsid nohup CMD </dev/null > /tmp/x.log 2>&1 &"`. Обычный `nohup CMD &` вешает ssh-сессию.
2. В однострочниках по ssh — `./venv/bin/pip` и `./venv/bin/python`, не `source activate`.
3. Каждой модели — Modelfile с `num_ctx` до первого запроса (волна 4).

---

## Волна 0. CLI + SSH + порт

```bash
tnr status --json --no-wait    # взять uuid; перед JSON есть строка "Fetching instances..."
timeout 20 script -q -c "tnr connect 0 -y" /dev/null
tnr ports forward 0 --add 80 --json
ssh tnr-0 "echo OK && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
```

Сайт: `https://<uuid>-80.thundercompute.net`

**Проброс порта — обязателен, и его забывают.** Без него всё внутри инстанса отвечает `200`, а снаружи Cloudflare отдаёт 404 «Nothing running here». Это единственное, что отделяло рабочий деплой от нерабочего 2026-07-30.

### Если `tnr` на машине нет (например, вы внутри инстанса)

Проброс делается и изнутри — CLI подхватывает `~/.thunder/token` сам:

```bash
# pip install tnr ставит МЁРТВУЮ версию 1.7.x: не знает ни --json, ни ports forward,
# её API-хост отвечает 404 на /min_version. Нужен бинарник с GitHub.
curl -s https://api.github.com/repos/Thunder-Compute/thunder-cli/releases/latest \
  | grep -E '"tag_name"'                       # на 2026-07-30 → v2.0.71
curl -sL -o tnr.tar.gz https://github.com/Thunder-Compute/thunder-cli/releases/download/v2.0.71/tnr_2.0.71_linux_amd64.tar.gz
tar xzf tnr.tar.gz && ./tnr --version

./tnr status --json --no-wait     # uuid лежит в поле "uuid"
./tnr ports forward 0 --add 80 --json   # → {"http_ports":[80]}, работает сразу, ждать не надо
```

### uuid без всякого CLI

`uuid` = `deviceId` из `~/.thunder/config.json` = `hostname` без префикса `instance-` и суффикса `-main`. Проверено: `tnr status` вернул ровно его.

**По 404 нельзя понять, в чём дело.** Ответ для несуществующего uuid (`https://zzzzzzzz-80.thundercompute.net`) байт-в-байт совпадает с ответом для верного uuid без проброса — те же 262 байта «Nothing running here». Так что 404 ≠ «неверный uuid»; сначала проверяйте проброс.

Если `tnr` ругается на версию — автообновление `/usr/bin/tnr` падает без sudo; запустить копию бинарника вне системных путей, она обновится сама, потом `sudo cp` в `/usr/bin`. При мёртвом токене `tnr login` врёт «Already logged in» — нужен `tnr logout` перед ним.

---

## Волна 1. Репозиторий + пакеты

```bash
ssh tnr-0 "git clone -q https://github.com/sherzod4033/Aigov.git ~/Aigov"   # пропустить, если ~/Aigov уже есть

ssh tnr-0 "sudo apt-get update -q >/dev/null 2>&1 && \
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
  nginx postgresql postgresql-contrib pciutils lshw \
  python3-venv python3-dev build-essential >/dev/null 2>&1 && \
  echo APT_OK && ls /usr/lib/postgresql/"
```

`pciutils`+`lshw` — иначе Ollama не найдёт GPU. Вывод `ls` в конце — версия PG для волны 3b.

---

## Волна 2. Node.js + Ollama

```bash
ssh tnr-0 "curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - >/dev/null 2>&1 && \
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q nodejs >/dev/null 2>&1 && node -v"

ssh tnr-0 "curl -fsSL https://ollama.com/install.sh | sh"
# нужно ">>> NVIDIA GPU installed."; "systemd is not running" — норма
```

---

## Волна 3. Сервисы + сборка

### 3a. Ollama + модели

```bash
ssh tnr-0 "mkdir -p ~/scripts && cat > ~/scripts/start_ollama.sh << 'EOF'
#!/bin/bash
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_NUM_PARALLEL=5
exec ollama serve
EOF
chmod +x ~/scripts/start_ollama.sh"

ssh -n tnr-0 "setsid nohup ~/scripts/start_ollama.sh </dev/null > /tmp/ollama.log 2>&1 &"
ssh tnr-0 "sleep 6; curl -s http://localhost:11434/api/tags"

ssh -n tnr-0 "setsid nohup ollama pull qwen3-embedding:8b </dev/null > /tmp/pull_qwen3.log 2>&1 &
setsid nohup ollama pull gemma4:26b </dev/null > /tmp/pull_g26.log 2>&1 &
setsid nohup ollama pull gemma4:e4b </dev/null > /tmp/pull_ge4b.log 2>&1 &
sleep 2; echo pulls_started"

# ожидание (~31 GB, 3–5 мин)
ssh tnr-0 "until ollama list | grep -q 'gemma4:26b'; do sleep 10; done; ollama list"
```

`OLLAMA_FLASH_ATTENTION=1` обязателен и только через `export` в скрипте — иначе "not enough system memory".

### 3b. PostgreSQL + ChromaDB

```bash
ssh tnr-0 "sudo -u postgres LC_ALL=C /usr/lib/postgresql/14/bin/pg_ctl start \
  -D /var/lib/postgresql/14/main \
  -o '-c config_file=/etc/postgresql/14/main/postgresql.conf' \
  -l /tmp/pg_ubuntu.log; sleep 3; \
  sudo -u postgres psql -c \"CREATE USER andozai_user WITH PASSWORD 'andozai_password';\"; \
  sudo -u postgres psql -c \"CREATE DATABASE andozai_db OWNER andozai_user;\""
# ";" а не "&&": pg_ctl отдаёт ненулевой код даже при успехе. Нативный PG — docker-образ падает

ssh tnr-0 "sudo docker run --name andozai-chromadb \
  -p 8000:8000 -v chroma_data:/chroma/chroma \
  -e IS_PERSISTENT=TRUE -e PERSIST_DIRECTORY=/chroma/chroma \
  -d chromadb/chroma:latest"
```

### 3c. Backend + Frontend

```bash
ssh tnr-0 "cd ~/Aigov/SafeDocsAI/backend && python3 -m venv venv && \
  ./venv/bin/pip install -q --upgrade pip && \
  ./venv/bin/pip install -q -r requirements.txt && echo BACKEND_DEPS_OK"

ssh tnr-0 "cd ~/Aigov/SafeDocsAI/frontend && npm install --silent && npm run build"
```

---

## Волна 4. Modelfile, запуск, nginx

```bash
ssh tnr-0 "printf 'FROM gemma4:e4b\nPARAMETER num_ctx 20000\n' > /tmp/Mf_e4b && ollama create gemma4:e4b -f /tmp/Mf_e4b"
ssh tnr-0 "printf 'FROM gemma4:26b\nPARAMETER num_ctx 12000\n' > /tmp/Mf_26b && ollama create gemma4:26b -f /tmp/Mf_26b"
ssh tnr-0 "printf 'FROM gemma4:31b\nPARAMETER num_ctx 12000\n' > /tmp/Mf_31b && ollama create gemma4:31b -f /tmp/Mf_31b"
```

Третья строка — про запас: волна 3a `gemma4:31b` не качает, и без `ollama pull` она упадёт. Нужна 31b — добавьте её в пуллы волны 3a.

> Без этого модель берёт дефолтный контекст 262144 → при `NUM_PARALLEL=5` KV-кэш 102 GB, половина слоёв уезжает в CPU. Симптом в `ollama ps`: `36%/64% CPU/GPU`.

### Backend

UUID из волны 0 подставить в `CORS_ORIGINS` — иначе фронтенд упрётся в CORS.

```bash
ssh tnr-0 "mkdir -p ~/scripts && cat > ~/scripts/start_backend.sh << 'SCRIPT'
#!/bin/bash
cd /home/ubuntu/Aigov/SafeDocsAI/backend
source venv/bin/activate
export POSTGRES_SERVER=localhost POSTGRES_PORT=5432
export POSTGRES_USER=andozai_user POSTGRES_PASSWORD=andozai_password POSTGRES_DB=andozai_db
export CHROMA_HOST=localhost CHROMA_PORT=8000
export OLLAMA_API_BASE=http://localhost:11434
export OLLAMA_MODEL_CHAT=gemma4:26b
export OLLAMA_MODEL_EMBEDDING=qwen3-embedding:8b
export ENVIRONMENT=production
# Ключ обязателен и проверяется на старте: плейсхолдеры и строки короче
# 32 символов отвергаются, бэкенд не поднимется. Сгенерировать один раз:
#   python3 -c "import secrets; print(secrets.token_hex(32))"
export SECRET_KEY=<64-символьный ключ>
export ALLOW_REGISTRATION=false
export CORS_ORIGINS=http://localhost:5173,http://localhost:3000,https://ПОДСТАВИТЬ_UUID-80.thundercompute.net
python -m app.init_db 2>&1 && python run.py 2>&1
SCRIPT
chmod +x ~/scripts/start_backend.sh"

ssh -n tnr-0 "setsid nohup ~/scripts/start_backend.sh </dev/null > /tmp/backend.log 2>&1 &"
ssh tnr-0 "sleep 22; tail -5 /tmp/backend.log"   # → "Uvicorn running on http://0.0.0.0:8001"
```

### Frontend + Nginx

```bash
ssh tnr-0 "sudo mkdir -p /var/www/aigov && sudo cp -r ~/Aigov/SafeDocsAI/frontend/dist/* /var/www/aigov/"

ssh tnr-0 "sudo tee /etc/nginx/sites-available/aigov > /dev/null << 'EOF'
server {
    listen 80 default_server;
    server_name _;
    root /var/www/aigov;
    index index.html;
    location / { try_files \$uri \$uri/ /index.html; }
    location /api/ {
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        # Без этой строки клиентский X-Forwarded-For доходит до бэкенда
        # нетронутым, и лимит на подбор пароля обходится сменой заголовка.
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
        proxy_send_timeout 300s;
        client_max_body_size 50M;
    }
}
EOF
sudo ln -sf /etc/nginx/sites-available/aigov /etc/nginx/sites-enabled/aigov
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo nginx"
# sudo nginx, не systemctl — systemd в K8s-поде нет
```

### Admin + проверка

```bash
ssh tnr-0 "cd ~/Aigov/SafeDocsAI/backend && \
  POSTGRES_SERVER=localhost POSTGRES_USER=andozai_user \
  POSTGRES_PASSWORD=andozai_password POSTGRES_DB=andozai_db \
  ADMIN_USERNAME=123 ADMIN_PASSWORD=123 ./venv/bin/python create_admin.py"

ssh tnr-0 "curl -s -o /dev/null -w 'nginx=%{http_code} ' http://localhost/
curl -s -o /dev/null -w 'backend=%{http_code} ' http://localhost:8001/
curl -s -o /dev/null -w 'chroma=%{http_code} ' http://localhost:8000/api/v2/heartbeat
curl -s -o /dev/null -w 'api=%{http_code}\n' http://localhost/api/v1/openapi.json
curl -s -X POST http://localhost/api/v1/auth/login \
  -H 'Content-Type: application/x-www-form-urlencoded' -d 'username=123&password=123'"
# всё 200 + JWT. Роута /api/v1/health не существует
```

Проверка чата на GPU:
```bash
ssh tnr-0 "curl -s --max-time 300 http://localhost:11434/api/generate \
  -d '{\"model\":\"gemma4:26b\",\"prompt\":\"Салом! Ту кистӣ?\",\"stream\":false}' | head -c 200
ollama ps"   # обязано быть 100% GPU
```

---

## `~/scripts/start_all.sh` — перезапуск после паузы

```bash
#!/bin/bash
sudo -u postgres LC_ALL=C /usr/lib/postgresql/14/bin/pg_ctl start -D /var/lib/postgresql/14/main \
  -o '-c config_file=/etc/postgresql/14/main/postgresql.conf' -l /tmp/pg_ubuntu.log
sudo docker start andozai-chromadb
pgrep -f 'ollama serve' >/dev/null || setsid nohup ~/scripts/start_ollama.sh </dev/null > /tmp/ollama.log 2>&1 &
sleep 5
pgrep -f 'nginx: master' >/dev/null || sudo nginx
pgrep -f 'run.py' >/dev/null || setsid nohup ~/scripts/start_backend.sh </dev/null > /tmp/backend.log 2>&1 &
sleep 8
curl -s -o /dev/null -w 'nginx=%{http_code} ' http://localhost/
curl -s -o /dev/null -w 'backend=%{http_code} ' http://localhost:8001/
curl -s -o /dev/null -w 'chroma=%{http_code}\n' http://localhost:8000/api/v2/heartbeat
```

После паузы меняются IP и UUID: сначала `tnr connect 0` + `tnr ports forward 0 --add 80`, потом поправить UUID в `CORS_ORIGINS`.

---

## Если что-то сломалось

**Сайт не открывается.** Сначала разделить «внутри» и «снаружи»:

```bash
curl -s -o /dev/null -w 'local=%{http_code}\n' http://localhost/
curl -s -o /dev/null -w 'public=%{http_code}\n' --max-time 20 https://<uuid>-80.thundercompute.net/
```

- `local=200`, `public=404` → порт не проброшен. `tnr ports forward 0 --add 80` (волна 0).
- оба 404 → nginx не поднят: `sudo nginx -t && sudo nginx`.
- сайт открылся, но запросы в UI падают → uuid в `CORS_ORIGINS` не тот. Поправить `~/scripts/start_backend.sh`, `pkill -f run.py`, перезапустить.

**Модель висит в `Stopping...` и держит VRAM** — `ollama stop` не помогает:
```bash
ssh tnr-0 "pkill -f 'ollama serve'; pkill -f 'ollama runner'; sleep 4"
ssh -n tnr-0 "setsid nohup ~/scripts/start_ollama.sh </dev/null > /tmp/ollama.log 2>&1 &"
```

**Зависшие документы** — руками больше не нужно. Индексация ушла в фоновую очередь (таблица `job`), задача берётся атомарно через `FOR UPDATE SKIP LOCKED` и держится на аренде с heartbeat. Если процесс убили посреди индексации, аренда протухает и воркер сам возвращает задачу в очередь: документ доходит до `indexed` примерно за 70-90 секунд после рестарта. Повторная обработка чистит старые чанки, дублей не возникает.

Посмотреть очередь:
```bash
sudo -u postgres psql -d andozai_db -c \
  "SELECT j.id, j.status, j.attempt_count, d.name, d.status, d.error_text
     FROM job j LEFT JOIN document d ON d.id = j.source_id
    ORDER BY j.created_at DESC LIMIT 20;"
```
Причина ошибки теперь лежит в `document.error_text` и отдаётся в API вместе со статусом.

**VRAM (замерено):** 26b — 22 GB, 31b — 37 GB, embedding — 5 GB, reranker — ~8 GB. Reranker и contextual chunking по дефолту выключены в `runtime_settings.py`, включаются в UI.
