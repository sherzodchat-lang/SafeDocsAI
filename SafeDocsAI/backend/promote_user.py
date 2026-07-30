import asyncio
from sqlmodel import select
from app.core.database import get_session
from app.models.models import User

async def promote_user(username: str):
    # Manually get a session since we are outside of FastAPI dependency injection
    async for session in get_session():
        statement = select(User).where(User.username == username)
        result = await session.exec(statement)
        user = result.first()
        
        if not user:
            print(f"User '{username}' not found.")
            return
        
        if user.role == "admin":
            print(f"User '{username}' is already an admin.")
            return

        user.role = "admin"
        session.add(user)
        await session.commit()
        await session.refresh(user)
        print(f"User '{username}' successfully promoted to admin.")
        return

if __name__ == "__main__":
    # You can change 'admin' to the actual username if it's different
    username_to_promote = "admin" 
    asyncio.run(promote_user(username_to_promote))
