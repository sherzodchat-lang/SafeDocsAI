import asyncio
from app.core.database import get_session
from app.models.models import User
from app.core.security import get_password_hash
from sqlmodel import select

async def create_user():
    async for session in get_session():
        # Check if user exists
        result = await session.exec(select(User).where(User.username == "testuser"))
        user = result.first()
        if user:
            print("User testuser already exists")
            return

        new_user = User(
            username="testuser",
            password_hash=get_password_hash("testpass123"),
            role="user"
        )
        session.add(new_user)
        await session.commit()
        print("User testuser created")

if __name__ == "__main__":
    asyncio.run(create_user())
