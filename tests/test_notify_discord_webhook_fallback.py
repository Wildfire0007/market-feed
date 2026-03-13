from unittest import mock

from scripts import notify_discord


class _Resp:
    def __init__(self, code: int):
        self.status_code = code


def test_send_discord_embed_tries_backup_webhook_and_succeeds():
    embed = {"title": "teszt"}
    requests_mock = mock.Mock()
    with mock.patch.object(notify_discord, "DRY_RUN", False), \
         mock.patch.object(notify_discord, "requests", requests_mock), \
         mock.patch.object(notify_discord, "DISCORD_WEBHOOK_URLS", ["https://bad", "https://good"]), \
         mock.patch.object(notify_discord, "NOTIFY_ATTEMPTS", 0), \
         mock.patch.object(notify_discord, "NOTIFY_SUCCESSES", 0), \
         mock.patch.object(notify_discord, "NOTIFY_FAILURES", 0), \
         mock.patch("scripts.notify_discord._append_notify_event") as event_log:
        requests_mock.post.side_effect = [_Resp(403), _Resp(204)]

        ok = notify_discord.send_discord_embed(embed)

        assert requests_mock.post.call_count == 2
        assert event_log.call_count == 2
        assert notify_discord.NOTIFY_ATTEMPTS == 1
        assert notify_discord.NOTIFY_SUCCESSES == 1
        assert notify_discord.NOTIFY_FAILURES == 0

    assert ok is True


def test_collect_webhook_urls_strips_and_deduplicates(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", " https://a ,\nhttps://b, https://b , https://c ")

    urls = notify_discord._collect_webhook_urls()

    assert urls == ["https://a", "https://b", "https://c"]


def test_send_discord_embed_single_webhook_content_fallback_on_403():
    embed = {"title": "teszt cím"}
    requests_mock = mock.Mock()
    with mock.patch.object(notify_discord, "DRY_RUN", False), \
         mock.patch.object(notify_discord, "requests", requests_mock), \
         mock.patch.object(notify_discord, "DISCORD_WEBHOOK_URLS", ["https://only"]), \
         mock.patch.object(notify_discord, "NOTIFY_ATTEMPTS", 0), \
         mock.patch.object(notify_discord, "NOTIFY_SUCCESSES", 0), \
         mock.patch.object(notify_discord, "NOTIFY_FAILURES", 0), \
         mock.patch("scripts.notify_discord._append_notify_event") as event_log:
        requests_mock.post.side_effect = [_Resp(403), _Resp(204)]

        ok = notify_discord.send_discord_embed(embed)

        assert ok is True
        assert requests_mock.post.call_count == 2
        first_call = requests_mock.post.call_args_list[0]
        second_call = requests_mock.post.call_args_list[1]
        assert first_call.kwargs["json"] == {"embeds": [embed]}
        assert second_call.kwargs["json"] == {"content": "teszt cím"}
        assert notify_discord.NOTIFY_ATTEMPTS == 1
        assert notify_discord.NOTIFY_SUCCESSES == 1
        assert notify_discord.NOTIFY_FAILURES == 0
        assert event_log.call_count == 2
