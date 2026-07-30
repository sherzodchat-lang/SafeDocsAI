from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.models import Insight, Notebook


class InsightsService:
    @staticmethod
    async def list_insights(
        session: AsyncSession,
        notebook_id: int | None = None,
        owner_id: int | None = None,
    ) -> list[Insight]:
        statement = select(Insight).order_by(Insight.updated_at.desc())
        if notebook_id is not None:
            statement = statement.where(Insight.notebook_id == notebook_id)
        # owner_id=None — вызов от админа: отдаём всё. Иначе ограничиваем
        # инсайтами из блокнотов пользователя: у Insight владельца нет,
        # источник истины — Notebook.owner_id.
        if owner_id is not None:
            statement = statement.where(
                Insight.notebook_id.in_(
                    select(Notebook.id).where(Notebook.owner_id == owner_id)
                )
            )
        result = await session.exec(statement)
        return result.all()
