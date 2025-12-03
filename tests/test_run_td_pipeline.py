import importlib

import scripts.run_td_pipeline as pipeline


def test_trading_available_accepts_td_api_key(monkeypatch):
    monkeypatch.delenv("TWELVEDATA_API_KEY", raising=False)
    monkeypatch.setenv("TD_API_KEY", "dummy")
    importlib.reload(pipeline)

    assert pipeline._trading_available(force=False) is True


def test_trading_available_false_without_keys(monkeypatch):
    monkeypatch.delenv("TWELVEDATA_API_KEY", raising=False)
    monkeypatch.delenv("TD_API_KEY", raising=False)
    importlib.reload(pipeline)

    assert pipeline._trading_available(force=False) is False


def test_news_available_uses_helper(monkeypatch):
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)
    importlib.reload(pipeline)

    assert pipeline._news_available(force=False) is False

    monkeypatch.setenv("NEWSAPI_KEY", "present")
    importlib.reload(pipeline)
    assert pipeline._news_available(force=False) is True
