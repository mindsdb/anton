import io

from anton.cloud_turn import session as cloud_session


def _real_png_bytes() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (2, 2), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def test_no_attachments_returns_plain_string(tmp_path):
    assert cloud_session.build_turn_content(tmp_path, "hello") == "hello"


def test_image_attachment_becomes_a_vision_block(tmp_path):
    att = tmp_path / "attachments" / "fileid1"
    att.mkdir(parents=True)
    (att / "shot.png").write_bytes(_real_png_bytes())

    content = cloud_session.build_turn_content(tmp_path, "do you see the screenshot?")
    assert isinstance(content, list)
    images = [b for b in content if b["type"] == "image"]
    assert len(images) == 1
    assert images[0]["source"]["media_type"] == "image/png"
    text = next(b for b in content if b["type"] == "text")["text"]
    assert "shot.png" in text and "do you see the screenshot?" in text


def test_non_image_is_listed_by_path_not_inlined(tmp_path):
    att = tmp_path / "attachments" / "fileid2"
    att.mkdir(parents=True)
    (att / "notes.txt").write_text("hi")

    content = cloud_session.build_turn_content(tmp_path, "read my notes")
    assert isinstance(content, list)
    assert not [b for b in content if b["type"] == "image"]
    text = next(b for b in content if b["type"] == "text")["text"]
    assert str(att / "notes.txt") in text            # absolute path so the agent can read it
    assert "read my notes" in text


def test_oversized_image_is_skipped_not_inlined(tmp_path, monkeypatch):
    import anton.utils.clipboard as clip
    monkeypatch.setattr(clip, "MAX_IMAGE_BYTES", 8)     # any real image exceeds this
    att = tmp_path / "attachments" / "big"
    att.mkdir(parents=True)
    (att / "huge.png").write_bytes(_real_png_bytes())

    content = cloud_session.build_turn_content(tmp_path, "look")
    assert isinstance(content, list)
    assert not [b for b in content if b["type"] == "image"]   # too large → no vision block
    assert "could not be shown" in next(b for b in content if b["type"] == "text")["text"]


def test_corrupt_image_degrades_to_listing_not_a_broken_block(tmp_path):
    """A mislabeled/corrupt .png must NOT be inlined (the model would reject it
    and fail every turn) — it degrades to a path listing line."""
    att = tmp_path / "attachments" / "bad"
    att.mkdir(parents=True)
    (att / "broken.png").write_bytes(b"\x89PNG\r\n not really a png")

    content = cloud_session.build_turn_content(tmp_path, "look")
    assert isinstance(content, list)
    assert not [b for b in content if b["type"] == "image"]
    assert "could not be shown" in next(b for b in content if b["type"] == "text")["text"]


def test_one_bad_image_does_not_drop_the_good_ones(tmp_path):
    att = tmp_path / "attachments"
    (att / "good").mkdir(parents=True)
    (att / "good" / "ok.png").write_bytes(_real_png_bytes())
    (att / "bad").mkdir(parents=True)
    (att / "bad" / "broken.png").write_bytes(b"nope")

    content = cloud_session.build_turn_content(tmp_path, "look")
    images = [b for b in content if b["type"] == "image"]
    assert len(images) == 1        # the good one survives the bad one


def _real_bmp_bytes() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (2, 2), (1, 2, 3)).save(buf, format="BMP")
    return buf.getvalue()


def test_bmp_bytes_named_png_are_not_shipped_as_png(tmp_path):
    """Format comes from content, not extension: a BMP named .png must be
    converted (via Pillow, present here), never shipped as raw BMP labeled png."""
    att = tmp_path / "attachments" / "id"
    att.mkdir(parents=True)
    (att / "scan.png").write_bytes(_real_bmp_bytes())   # BMP bytes, .png name

    content = cloud_session.build_turn_content(tmp_path, "look")
    images = [b for b in content if b["type"] == "image"]
    assert len(images) == 1
    assert images[0]["source"]["media_type"] == "image/png"       # converted, correct type
    import base64
    assert base64.standard_b64decode(images[0]["source"]["data"]).startswith(b"\x89PNG")  # real PNG bytes


def test_png_works_without_pillow_bmp_degrades(tmp_path, monkeypatch):
    """Pillow is optional in the prod image. PNG/JPEG/etc must inline with no
    Pillow; a BMP (needs Pillow to convert) degrades to a listing line."""
    png = _real_png_bytes()
    bmp = _real_bmp_bytes()
    att = tmp_path / "attachments" / "id"
    att.mkdir(parents=True)
    (att / "a.png").write_bytes(png)
    (att / "b.bmp").write_bytes(bmp)

    import builtins
    real_import = builtins.__import__

    def no_pil(name, *a, **k):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("simulating the Pillow-less prod image")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_pil)
    content = cloud_session.build_turn_content(tmp_path, "look")
    images = [b for b in content if b["type"] == "image"]
    assert len(images) == 1                                        # PNG inlined without Pillow
    assert images[0]["source"]["media_type"] == "image/png"
    text = next(b for b in content if b["type"] == "text")["text"]
    assert "b.bmp" in text                                         # BMP degraded to listing
