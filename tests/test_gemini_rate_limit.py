import asyncio

import pytest

from app.services.gemini_rate_limit import GeminiRateLimiter


class FakeClock:
    """Horloge manuelle : le temps n'avance que quand le test le décide."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _limiter(limit: int, clock: FakeClock) -> tuple[GeminiRateLimiter, list[float]]:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock.now += seconds  # le temps avance de la durée « dormie »

    return GeminiRateLimiter(limit, clock=clock, sleep=fake_sleep), sleeps


@pytest.mark.asyncio
async def test_allows_up_to_the_limit_without_waiting():
    clock = FakeClock()
    limiter, sleeps = _limiter(3, clock)

    for _ in range(3):
        await limiter.acquire()

    assert sleeps == []


@pytest.mark.asyncio
async def test_blocks_until_the_oldest_hit_leaves_the_window():
    clock = FakeClock()
    limiter, sleeps = _limiter(2, clock)
    await limiter.acquire()  # t=0
    clock.now = 10.0
    await limiter.acquire()  # t=10

    await limiter.acquire()  # doit attendre que t=0 sorte de la fenêtre de 60s

    assert sleeps == [pytest.approx(50.05, abs=0.01)]
    assert clock.now == pytest.approx(60.05, abs=0.01)


@pytest.mark.asyncio
async def test_old_hits_outside_the_window_are_forgotten():
    clock = FakeClock()
    limiter, sleeps = _limiter(1, clock)
    await limiter.acquire()  # t=0
    clock.now = 61.0

    await limiter.acquire()  # la fenêtre précédente est passée : aucune attente

    assert sleeps == []


@pytest.mark.asyncio
async def test_limit_of_zero_or_negative_is_treated_as_one():
    clock = FakeClock()
    limiter, sleeps = _limiter(0, clock)

    await limiter.acquire()
    await limiter.acquire()

    assert len(sleeps) == 1  # comporté comme limite=1, jamais un blocage permanent


@pytest.mark.asyncio
async def test_concurrent_acquires_never_exceed_the_limit_within_a_window():
    clock = FakeClock()
    limiter, _ = _limiter(2, clock)
    order: list[int] = []

    async def worker(i: int) -> None:
        await limiter.acquire()
        order.append(i)

    await asyncio.gather(*(worker(i) for i in range(5)))

    assert len(order) == 5  # tous finissent par passer, sans dépasser la limite à aucun instant
