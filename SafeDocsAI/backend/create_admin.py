"""Create or update a default admin account."""

import asyncio
import os

from sqlmodel import select

from app.core.database import get_session
from app.core.security import get_password_hash
from app.models.models import User

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")


async def create_admin() -> None:
    async for session in get_session():
        result = await session.exec(select(User).where(User.username == ADMIN_USERNAME))
        user = result.first()

        if user:
            # Намеренно ничего не меняем: повторный прогон деплоя не должен
            # сбрасывать пароль действующего пользователя и повышать его роль.
            print(
                f"User '{ADMIN_USERNAME}' already exists (role: {user.role}) — left untouched."
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
