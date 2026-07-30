from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.models import Note, Notebook


class NotesService:
    @staticmethod
    async def list_notes(
        session: AsyncSession,
        notebook_id: int | None = None,
        owner_id: int | None = None,
    ) -> list[Note]:
        statement = select(Note).order_by(Note.updated_at.desc())
        if notebook_id is not None:
            statement = statement.where(Note.notebook_id == notebook_id)
        # owner_id=None — вызов от админа: отдаём всё. Иначе ограничиваем
        # заметками из блокнотов пользователя: у Note владельца нет,
        # источник истины — Notebook.owner_id.
        if owner_id is not None:
            statement = statement.where(
                Note.notebook_id.in_(
                    select(Notebook.id).where(Notebook.owner_id == owner_id)
                )
            )
        result = await session.exec(statement)
        return result.all()
