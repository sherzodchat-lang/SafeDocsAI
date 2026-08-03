"""Create or update a default admin account.

Заодно это аварийный вход (break-glass): единственный способ вернуть роль
admin, когда админов в системе не осталось. Через API это невозможно —
PUT /settings/users/{id}/role сам требует роль admin, создания пользователя в
API нет, а регистрация жёстко ставит role="user" и по умолчанию выключена.
Порядок действий записан в DEPLOY.md, раздел «Если что-то сломалось».
"""

import asyncio
import os

from sqlmodel import select

from app.core.database import get_session
from app.core.security import get_password_hash
from app.models.models import User

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
# Повысить существующего пользователя до admin. По умолчанию выключено: см.
# ветку ниже — обычный прогон деплоя не должен менять ничьи права.
ADMIN_PROMOTE = os.getenv("ADMIN_PROMOTE", "").strip().lower() in ("1", "true", "yes")


async def create_admin() -> None:
    async for session in get_session():
        result = await session.exec(select(User).where(User.username == ADMIN_USERNAME))
        user = result.first()

        if user:
            # Намеренно ничего не меняем: повторный прогон деплоя не должен
            # сбрасывать пароль действующего пользователя и повышать его роль.
            if not ADMIN_PROMOTE:
                print(
                    f"User '{ADMIN_USERNAME}' already exists (role: {user.role}) — left untouched. "
                    f"Set ADMIN_PROMOTE=1 to grant this user the admin role."
                )
                return

            # ADMIN_PROMOTE=1 — осознанное исключение, аварийный путь. Пароль
            # не трогаем и здесь: возвращать доступ нужно тому, кто им уже
            # владеет, а не подменять его учётные данные.
            if user.role == "admin":
                print(f"User '{ADMIN_USERNAME}' is already an admin — nothing to do.")
                return

            previous_role = user.role
            user.role = "admin"
            session.add(user)
            await session.commit()
            print(
                f"Promoted user '{ADMIN_USERNAME}': {previous_role} -> admin. "
                f"Password left unchanged."
            )
            return
        else:
            user = User(
                username=ADMIN_USERNAME,
                password_hash=get_password_hash(ADMIN_PASSWORD),
                role="admin",
            )
            session.add(user)
            await session.commit()
            print(f"Created new admin user '{ADMIN_USERNAME}'.")

    print(f"Username: {ADMIN_USERNAME}")
    print(f"Password: {ADMIN_PASSWORD}")


if __name__ == "__main__":
    asyncio.run(create_admin())
