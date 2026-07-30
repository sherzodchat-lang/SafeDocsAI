import asyncio
from sqlmodel import select
from app.core.database import get_session
from app.models.models import User

async def list_users():
    async for session in get_session():
        statement = select(User)
        result = await session.exec(statement)
        users = result.all()
        
        print(f"{'Username':<20} | {'Role':<10} | {'ID':<5}")
        print("-" * 40)
        for user in users:
            print(f"{user.username:<20} | {user.role:<10} | {user.id:<5}")

if __name__ == "__main__":
    asyncio.run(list_users())
