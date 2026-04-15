import json
import urllib.request
from pathlib import Path
from xml.dom import minidom
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf2bpmn.processgpt.bpmn_xml_generator import ProcessGPTBPMNXmlGenerator


def read_env(path: Path) -> dict[str, str]:
    kv: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        kv[k.strip()] = v.strip()
    return kv


def main() -> None:
    env = read_env(Path(".env"))
    base = env["SUPABASE_URL"]
    token = env["SERVICE_ROLE_KEY"]

    url = f"{base}/rest/v1/proc_def?id=eq.a44ef1a5-daee-4001-bee3-69f6c1655f5e&select=definition"
    req = urllib.request.Request(
        url,
        headers={
            "apikey": token,
            "Authorization": f"Bearer {token}",
        },
    )
    rows = json.loads(urllib.request.urlopen(req, timeout=20).read().decode("utf-8"))
    definition = ((rows[0] or {}).get("definition") if rows else None) or {}

    generator = ProcessGPTBPMNXmlGenerator()
    xml = generator.create_bpmn_xml(definition, horizontal=bool(definition.get("isHorizontal", True)))

    out = Path("output/a44_from_generator_fixed.bpmn")
    pretty = minidom.parseString(xml.encode("utf-8")).toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")
    pretty = "\n".join(line for line in pretty.splitlines() if line.strip())
    out.write_text(pretty, encoding="utf-8")

    print(f"out={out}")
    print(f"activities={len(definition.get('activities') or [])}")
    print(f"sequences={len(definition.get('sequences') or [])}")


if __name__ == "__main__":
    main()
