Отлично, вот полная пошаговая инструкция в том же порядке, как я делал.
Ниже я чуть упростил повторяющийся ssh через переменные (KEY и HOST) — это эквивалентно тем командам, которые я запускал с полным путем каждый раз.
KEY="/home/sherzod/Рабочий стол/ilm/ads-195947-and-2545.pem"
HOST="ubuntu@195.209.219.237"
1) Проверка ключа и подключение по SSH
ls -l "$KEY"
ssh -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -i "$KEY" "$HOST" "hostname && whoami && uname -a"
- На этом шаге был отказ: ключ имел слишком открытые права (0664).
Исправил права:
chmod 600 "$KEY"
ssh -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -i "$KEY" "$HOST" "hostname && whoami && uname -a"
---
2) Первичная диагностика сервера
ssh -i "$KEY" "$HOST" "set -e; echo '=== OS ==='; lsb_release -ds; echo '=== Tools ==='; command -v git || true; command -v docker || true; command -v docker-compose || true; command -v python3 || true; command -v pip3 || true; command -v node || true; command -v npm || true; command -v nginx || true; command -v ollama || true"
ssh -i "$KEY" "$HOST" "df -h / && free -h"
ssh -i "$KEY" "$HOST" "python3 --version && (pip3 --version || true)"
---
3) Установка системных зависимостей (Docker, Python, Nginx, OCR и т.д.)
ssh -i "$KEY" "$HOST" "sudo apt-get update && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io docker-compose-v2 nginx python3-pip python3-venv build-essential libpq-dev tesseract-ocr tesseract-ocr-rus poppler-utils curl"
Проверка версий:
ssh -i "$KEY" "$HOST" "set -e; docker --version; docker compose version; python3 --version; pip3 --version; nginx -v"
Включение Docker и автозапуск:
ssh -i "$KEY" "$HOST" "sudo systemctl enable --now docker && sudo usermod -aG docker ubuntu && systemctl is-active docker"
---
4) Установка Node.js 20
ssh -i "$KEY" "$HOST" "node --version || true; npm --version || true"
ssh -i "$KEY" "$HOST" "curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nodejs"
ssh -i "$KEY" "$HOST" "node --version && npm --version"
---
5) Установка Ollama + загрузка модели gemma3n:e4b
ssh -i "$KEY" "$HOST" "curl -fsSL https://ollama.com/install.sh | sh"
ssh -i "$KEY" "$HOST" "ollama --version && systemctl is-active ollama && systemctl status --no-pager --lines=20 ollama"
ssh -i "$KEY" "$HOST" "ollama pull gemma3n:e4b"
ssh -i "$KEY" "$HOST" "ollama list"
---
6) Клонирование проекта
ssh -i "$KEY" "$HOST" "ls -la /home/ubuntu"
ssh -i "$KEY" "$HOST" "git clone https://github.com/sherzod4033/Aigov.git /home/ubuntu/Aigov"
ssh -i "$KEY" "$HOST" "ls -la /home/ubuntu/Aigov"
---
7) Поднятие инфраструктуры проекта (Postgres + ChromaDB)
ssh -i "$KEY" "$HOST" "sudo docker compose -f /home/ubuntu/Aigov/soliqai/docker-compose.yml up -d postgres chromadb"
ssh -i "$KEY" "$HOST" "sudo docker compose -f /home/ubuntu/Aigov/soliqai/docker-compose.yml ps"
ssh -i "$KEY" "$HOST" "sudo docker update --restart unless-stopped soliqai-postgres soliqai-chromadb"
---
8) Backend: venv + python-зависимости
ssh -i "$KEY" "$HOST" "python3 -m venv /home/ubuntu/Aigov/soliqai/backend/venv && /home/ubuntu/Aigov/soliqai/backend/venv/bin/pip install --upgrade pip && /home/ubuntu/Aigov/soliqai/backend/venv/bin/pip install -r /home/ubuntu/Aigov/soliqai/backend/requirements.txt"
---
9) Создание backend .env
Проверил, что .env отсутствует:
ssh -i "$KEY" "$HOST" "ls -l /home/ubuntu/Aigov/soliqai/backend/.env /home/ubuntu/Aigov/soliqai/backend/.env.example"
Создал .env (с сгенерированным SECRET_KEY):
ssh -i "$KEY" "$HOST" "python3 - <<'PY'
from pathlib import Path
import secrets
secret = secrets.token_hex(32)
content = f'''OPENAI_API_KEY=
OPENAI_MODEL=gpt-4.1-mini
POSTGRES_USER=soliqai_user
POSTGRES_PASSWORD=soliqai_password
POSTGRES_SERVER=localhost
POSTGRES_PORT=5432
POSTGRES_DB=soliqai_db
SECRET_KEY={secret}
ACCESS_TOKEN_EXPIRE_MINUTES=30
CHROMA_HOST=localhost
CHROMA_PORT=8000
CHROMA_PERSIST_DIR=data/chroma
CORS_ORIGINS=http://195.209.219.237,http://localhost,http://127.0.0.1
ENVIRONMENT=production
'''
Path('/home/ubuntu/Aigov/soliqai/backend/.env').write_text(content, encoding='utf-8')
PY"
ssh -i "$KEY" "$HOST" "chmod 600 /home/ubuntu/Aigov/soliqai/backend/.env"
---
10) Инициализация таблиц БД
ssh -i "$KEY" "$HOST" "cd /home/ubuntu/Aigov/soliqai/backend && /home/ubuntu/Aigov/soliqai/backend/venv/bin/python -m app.init_db"
---
11) Frontend: установка npm-зависимостей и production build
ssh -i "$KEY" "$HOST" "cd /home/ubuntu/Aigov/soliqai/frontend && npm ci && VITE_API_BASE_URL=/api/v1 npm run build"
---
12) Публикация фронта в /var/www/soliqai
ssh -i "$KEY" "$HOST" "ls -la /var/www"
ssh -i "$KEY" "$HOST" "sudo mkdir -p /var/www/soliqai && sudo cp -a /home/ubuntu/Aigov/soliqai/frontend/dist/. /var/www/soliqai/"
---
13) Backend как systemd-сервис
ssh -i "$KEY" "$HOST" "sudo tee /etc/systemd/system/soliqai-backend.service > /dev/null <<'EOF'
[Unit]
Description=SoliqAI FastAPI Backend
After=network-online.target docker.service
Wants=network-online.target
Requires=docker.service
[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/Aigov/soliqai/backend
Environment=PATH=/home/ubuntu/Aigov/soliqai/backend/venv/bin:/usr/bin
ExecStart=/home/ubuntu/Aigov/soliqai/backend/venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8001 --workers 2
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload && sudo systemctl enable --now soliqai-backend && systemctl is-active soliqai-backend"
Проверка backend напрямую:
ssh -i "$KEY" "$HOST" "curl -fsS http://127.0.0.1:8001/health && echo && curl -fsS http://127.0.0.1:8001/ready"
---
14) Настройка Nginx (статика + прокси /api)
Сделал конфиг и включил сайт:
ssh -i "$KEY" "$HOST" "sudo tee /etc/nginx/sites-available/soliqai > /dev/null <<'EOF'
server {
    listen 80;
    server_name 195.209.219.237 _;
    root /var/www/soliqai;
    index index.html;
    client_max_body_size 50M;
    location /api/ {
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    location = /health {
        proxy_pass http://127.0.0.1:8001/health;
        proxy_set_header Host \$host;
    }
    location / {
        try_files \$uri \$uri/ /index.html;
    }
}
EOF
sudo ln -sf /etc/nginx/sites-available/soliqai /etc/nginx/sites-enabled/soliqai && sudo rm -f /etc/nginx/sites-enabled/default && sudo nginx -t && sudo systemctl restart nginx && systemctl is-active nginx"
---
15) Финальные проверки доступности
curl -fsSI http://195.209.219.237
curl -fsS http://195.209.219.237/health
curl -fsS http://195.209.219.237/api/v1/openapi.json | python3 -c "import sys,json; data=json.load(sys.stdin); print(data.get('info',{}).get('title','')); print(len(data.get('paths',{})))"
ssh -i "$KEY" "$HOST" "ollama run gemma3n:e4b 'Reply with only: ready'"
---
16) Дополнительные проверки сервисов
ssh -i "$KEY" "$HOST" "systemctl is-active docker ollama soliqai-backend nginx"
ssh -i "$KEY" "$HOST" "sudo docker ps --format '{{.Names}}\t{{.Status}}\t{{.Ports}}'"
ssh -i "$KEY" "$HOST" "curl -fsS http://127.0.0.1/api/v1/openapi.json >/dev/null && echo ok"
ssh -i "$KEY" "$HOST" "sudo journalctl -u soliqai-backend --no-pager -n 30"
ssh -i "$KEY" "$HOST" "if [ -f /var/run/reboot-required ]; then cat /var/run/reboot-required; else echo no-reboot-required; fi"
ssh -i "$KEY" "$HOST" "sudo ufw status"
---
17) Добавление admin-пользователя
Сначала проверил список пользователей:
ssh -i "$KEY" "$HOST" "cd /home/ubuntu/Aigov/soliqai/backend && /home/ubuntu/Aigov/soliqai/backend/venv/bin/python list_users.py"
Создал (или обновил) пользователя admin с ролью admin:
ssh -i "$KEY" "$HOST" "cd /home/ubuntu/Aigov/soliqai/backend && /home/ubuntu/Aigov/soliqai/backend/venv/bin/python - <<'PY'
import asyncio
import secrets
import string
from sqlmodel import select
from app.core.database import get_session
from app.models.models import User
from app.core.security import get_password_hash
USERNAME = 'admin'
ALPHABET = string.ascii_letters + string.digits
password = ''.join(secrets.choice(ALPHABET) for _ in range(18))
async def main():
    async for session in get_session():
        result = await session.exec(select(User).where(User.username == USERNAME))
        user = result.first()
        if user:
            user.role = 'admin'
            user.password_hash = get_password_hash(password)
            session.add(user)
            await session.commit()
            await session.refresh(user)
            print(f'UPDATED username={user.username} id={user.id} role={user.role} password={password}')
            return
        new_user = User(
            username=USERNAME,
            password_hash=get_password_hash(password),
            role='admin',
        )
        session.add(new_user)
        await session.commit()
        await session.refresh(new_user)
        print(f'CREATED username={new_user.username} id={new_user.id} role={new_user.role} password={password}')
        return
asyncio.run(main())
PY"
Проверил логин:
curl -fsS -X POST "http://195.209.219.237/api/v1/auth/login/access-token" -H "Content-Type: application/x-www-form-urlencoded" -d "username=admin&password=1TrW1fTsiFnW6qoF1Q"
Проверил admin-доступ:
curl -fsS "http://195.209.219.237/api/v1/settings/users" -H "Authorization: Bearer <TOKEN_ИЗ_ПРЕДЫДУЩЕЙ_КОМАНДЫ>"
---
18) Обязательные шаги после каждого git pull
Обновить код:
ssh -i "$KEY" "$HOST" "cd /home/ubuntu/Aigov && git pull origin main"
Применить обновления схемы БД и таблиц:
ssh -i "$KEY" "$HOST" "cd /home/ubuntu/Aigov/soliqai/backend && /home/ubuntu/Aigov/soliqai/backend/venv/bin/python -m app.init_db"
Перезапустить backend и frontend:
ssh -i "$KEY" "$HOST" "sudo systemctl restart soliqai-backend nginx && systemctl is-active soliqai-backend nginx"
Быстрая проверка:
curl -fsS http://195.209.219.237/health
curl -fsS http://195.209.219.237/api/v1/openapi.json >/dev/null && echo api-ok
---
19) Если после обновления ошибка "column chunk_index does not exist"
Сначала запустить штатную инициализацию:
ssh -i "$KEY" "$HOST" "cd /home/ubuntu/Aigov/soliqai/backend && /home/ubuntu/Aigov/soliqai/backend/venv/bin/python -m app.init_db"
Если ошибка не ушла, выполнить SQL вручную:
ssh -i "$KEY" "$HOST" "sudo docker exec -i soliqai-postgres psql -U soliqai_user -d soliqai_db -c \"ALTER TABLE chunk ADD COLUMN IF NOT EXISTS chunk_index INTEGER;\""
ssh -i "$KEY" "$HOST" "sudo docker exec -i soliqai-postgres psql -U soliqai_user -d soliqai_db -c \"WITH ranked AS (SELECT id, ROW_NUMBER() OVER (PARTITION BY doc_id ORDER BY id) - 1 AS rn FROM chunk) UPDATE chunk c SET chunk_index = ranked.rn FROM ranked WHERE c.id = ranked.id AND c.chunk_index IS NULL;\""
После этого перезапуск:
ssh -i "$KEY" "$HOST" "sudo systemctl restart soliqai-backend && systemctl is-active soliqai-backend"
---
20) Настройка таймаутов Nginx для больших PDF/DOCX
Если загрузки падают по 504, в блоке location /api/ должны быть:
proxy_connect_timeout 30s;
proxy_send_timeout 600s;
proxy_read_timeout 600s;
send_timeout 600s;
Проверка и перезагрузка:
ssh -i "$KEY" "$HOST" "sudo nginx -t && sudo systemctl reload nginx"
Проверка логов при ошибках загрузки:
ssh -i "$KEY" "$HOST" "sudo journalctl -u soliqai-backend --since '20 minutes ago' --no-pager"
ssh -i "$KEY" "$HOST" "python3 - <<'PY'
from pathlib import Path
for ln in Path('/var/log/nginx/access.log').read_text(errors='ignore').splitlines()[-30:]:
    print(ln)
PY"
---
21) Безопасность (обязательно)
НЕЛЬЗЯ хранить приватный ключ (.pem) в репозитории.
Если ключ попал в git:
git rm --cached ads-195947-and-2545.pem
echo "ads-195947-and-2545.pem" >> .gitignore
git add .gitignore
git commit -m "remove private key from git and ignore it"
git push
Сразу заменить ключ на сервере (rotate key pair).

Пароль admin лучше сменить на постоянный и не хранить в заметках.
Пример смены через API:
1) Войти под admin и получить токен.
2) Вызвать endpoint смены пароля (если включен в текущей версии), либо сменить через SQL/скрипт и перезапустить backend.
---
22) Быстрый чек-лист состояния прода
Проверка сервисов:
ssh -i "$KEY" "$HOST" "systemctl is-active docker ollama soliqai-backend nginx"
Проверка контейнеров:
ssh -i "$KEY" "$HOST" "sudo docker ps --format '{{.Names}}\t{{.Status}}\t{{.Ports}}'"
Проверка API и UI:
curl -fsS http://195.209.219.237/health
curl -fsSI http://195.209.219.237
Проверка последних ошибок backend:
ssh -i "$KEY" "$HOST" "sudo journalctl -u soliqai-backend --no-pager -n 80"
