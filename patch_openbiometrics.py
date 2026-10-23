"""Apply small, checked production defaults to the pinned upstream entrypoint."""
from pathlib import Path

path = Path("/opt/openbiometrics/api/app/main.py")
source = path.read_text(encoding="utf-8")
replacements = {
    """            ctx_id=-1,  # CPU on Mac, set to 0 for GPU
        ),""": """            ctx_id=-1,
            enable_demographics=False,
            enable_quality=True,
        ),""",
    """        document=DocumentConfig(
            enabled=True,""": """        document=DocumentConfig(
            enabled=False,""",
    """        person=PersonConfig(
            enabled=True,""": """        person=PersonConfig(
            enabled=False,""",
    """        video=VideoConfig(
            enabled=True,""": """        video=VideoConfig(
            enabled=False,""",
    """        events=EventsConfig(
            enabled=True,
            webhooks_enabled=True,""": """        events=EventsConfig(
            enabled=False,
            webhooks_enabled=False,""",
}
for old, new in replacements.items():
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"Expected exactly one upstream config block, found {count}: {old[:60]!r}")
    source = source.replace(old, new, 1)
path.write_text(source, encoding="utf-8")
print("Applied minimal, CPU, no-demographics production configuration.")
