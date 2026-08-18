"""3-layer validation gate per the officially documented z.ai pattern
(json_object -> client-side jsonschema) plus domain checks (R parse, jarl lint).
Every reject is logged with its failure layer for prompt iteration."""
import json, subprocess, tempfile
from pathlib import Path

ANALYST_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "minLength": 10},
        "code": {"type": "string", "minLength": 40},
        "packages_used": {"type": "array", "items": {"type": "string"},
                          "minItems": 0, "maxItems": 12},
    },
    "required": ["intent", "code", "packages_used"],
    "additionalProperties": False,
}
FINISH_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "minLength": 10},
        "body": {"type": "string", "minLength": 20},
        "packages_used": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["intent", "body", "packages_used"],
    "additionalProperties": False,
}


def _check_jsonschema(obj, schema):
    # minimal structural check (no external dep): full jsonschema if available
    try:
        import jsonschema
        jsonschema.validate(obj, schema)
        return True
    except ImportError:
        return (isinstance(obj, dict) and
                all(k in obj for k in schema["required"]) and
                isinstance(obj.get("intent"), str))
    except Exception:
        return False


def _r_parse_ok(code):
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run(["Rscript", "-e", f"invisible(parse('{path}'))"],
                           capture_output=True, text=True, timeout=30)
        return r.returncode == 0, r.stderr[-200:] if r.returncode else ""
    except Exception as e:
        return False, str(e)[:200]
    finally:
        Path(path).unlink(missing_ok=True)


def _jarl(code):
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "snippet.R"
        f.write_text(code)
        r = subprocess.run(["jarl", "check", "--allow-no-vcs", str(f)],
                           capture_output=True, text=True, timeout=60)
        warns = r.stdout.count("`warning`") + r.stdout.count("`error`")
        return warns


def validate(obj, schema, code_key="code", run_jarl=True):
    """Returns (ok, layer, info, jarl_warnings). Layers: json, parse, jarl."""
    if not _check_jsonschema(obj, schema):
        return False, "json", "schema/structure fail", 0
    code = obj[code_key]
    ok, err = _r_parse_ok(code)
    if not ok:
        return False, "parse", err, 0
    if run_jarl:
        w = _jarl(code)
        # jarl warnings are informational for analyst snippets; hard-fail only on many
        if w >= 5:
            return False, "jarl", f"{w} warnings/errors", w
        return True, "ok", "", w
    return True, "ok", "", 0
