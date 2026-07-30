import asyncio
from app.core.database import init_db
from app.shared.models import *  # Import models to register them with SQLModel metadata


async def main():
    print("Creating tables...")
    await init_db()
    print("Tables created successfully.")


if __name__ == "__main__":
    asyncio.run(main())
