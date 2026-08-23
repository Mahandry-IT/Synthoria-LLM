from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings


def create_engine(settings: Settings):
    """Crée l'engine async et le sessionmaker à partir de la config."""
    engine = create_async_engine(
        settings.database_url,
        echo=False,
        pool_pre_ping=True,
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return engine, session_factory


async def get_db_session(session_factory: async_sessionmaker) -> AsyncGenerator[AsyncSession, None]:
    """Dependency FastAPI : fournit une session DB par requête."""
    async with session_factory() as session:
        yield session
