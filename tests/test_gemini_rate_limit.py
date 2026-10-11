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


# ─── Limite en tokens (TPM) ───────────────────────────────────


def _token_limiter(tpm_limits: dict[str, int], clock: FakeClock) -> tuple[GeminiRateLimiter, list[float]]:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock.now += seconds

    return GeminiRateLimiter(100, tpm_limits=tpm_limits, clock=clock, sleep=fake_sleep), sleeps


@pytest.mark.asyncio
async def test_tokens_within_the_budget_do_not_wait():
    clock = FakeClock()
    limiter, sleeps = _token_limiter({"flash": 1000}, clock)

    await limiter.acquire_tokens("flash", 400)
    await limiter.acquire_tokens("flash", 600)

    assert sleeps == []


@pytest.mark.asyncio
async def test_tokens_over_the_budget_wait_for_the_oldest_reservation_to_expire():
    clock = FakeClock()
    limiter, sleeps = _token_limiter({"flash": 1000}, clock)
    await limiter.acquire_tokens("flash", 700)  # t=0
    clock.now = 20.0

    await limiter.acquire_tokens("flash", 400)  # 700 + 400 > 1000 : attend que t=0 expire

    assert sleeps == [pytest.approx(40.05, abs=0.01)]


@pytest.mark.asyncio
async def test_token_budgets_are_independent_per_model():
    clock = FakeClock()
    limiter, sleeps = _token_limiter({"flash": 1000, "lite": 1000}, clock)

    await limiter.acquire_tokens("flash", 900)
    await limiter.acquire_tokens("lite", 900)

    assert sleeps == []


@pytest.mark.asyncio
async def test_model_without_token_limit_is_never_throttled():
    clock = FakeClock()
    limiter, sleeps = _token_limiter({"flash": 1000}, clock)

    assert limiter.has_token_limit("lite") is False
    assert await limiter.acquire_tokens("lite", 10_000_000) is None
    assert sleeps == []


@pytest.mark.asyncio
async def test_oversized_request_passes_alone_instead_of_blocking_forever():
    clock = FakeClock()
    limiter, sleeps = _token_limiter({"flash": 1000}, clock)

    await limiter.acquire_tokens("flash", 5000)  # fenêtre vide : passe malgré le dépassement

    assert sleeps == []

    await limiter.acquire_tokens("flash", 5000)  # fenêtre occupée : attend qu'elle se vide

    assert sleeps == [pytest.approx(60.05, abs=0.01)]


@pytest.mark.asyncio
async def test_corrected_reservation_frees_the_overestimated_tokens():
    clock = FakeClock()
    limiter, sleeps = _token_limiter({"flash": 1000}, clock)
    reservation = await limiter.acquire_tokens("flash", 900)

    limiter.correct_tokens(reservation, 300)
    await limiter.acquire_tokens("flash", 600)  # 300 + 600 tiennent, 900 + 600 n'auraient pas tenu

    assert sleeps == []
