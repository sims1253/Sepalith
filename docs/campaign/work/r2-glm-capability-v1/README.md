# R2 GLM capability preparation

Status: **blocked for the exact requested model**. The pinned catalog contains `glm-5.3` for
`zai`, `bigmodel`, `zai-coding-plan`, and `bigmodel-coding-plan`; it contains zero
`glm-5.3-flash` entries. A bounded exact-target launch must stop at this gate. The nearest
registered ID is `zai-coding-plan/glm-5.3`, but this artifact does not silently substitute it.

The packaged entrypoint is `/opt/ZCode/resources/glm/zcode.cjs` (0.16.5). Its source parser
accepts a runtime denylist (`--disallowed-tools`) and `--mode`, `--cwd`, `--prompt`, and
`--surface`; its observed parser does not accept help-advertised `--settings`, `--max-turns`,
`--allowed-tools`, or `--permission-mode`. Use the project config template plus denylist and
an external 600-second guard. The shell template is intentionally not executed here.

There are two different capabilities:

* **Native desktop computer-use:** the packaged `computer-use` plugin is a stdio MCP server
  (0.5.14) with read-only tools such as `list_apps`, `list_windows`, `get_app_state`, and
  `screenshot`; input tools are tier T1. It needs a host CUA broker. The package alone does
  not prove that the notebook host advertises or authorizes a desktop broker. Root must do a
  later read-only host capability check. The policy template leaves read-only CUA available
  and denies all input/mutation tools.
* **Headless browser-use:** `--browser-use=headless` creates a managed Chromium CDP backend
  when an executable is supplied or pinned. This is a browser capability, not native desktop
  CUA. The optional note records the flags; it is disabled in the CUA policy by default.

The diagnostic prompt is synthetic/read-only and must not read credentials, training/final data,
full app logs, or modify files. A host launch should verify the exact requested model in its
active provider registry before starting the bounded command. If that registry still lacks the
ID, report a model capability no-go rather than infer support from `glm-5.3`.

Files:

* `source-manifest.json` — exact source identities and bounded scan results.
* `cli-parser-evidence.txt` — compact parser/help evidence.
* `tool-policy.json` — denylist and host checks.
* `zcode-project-config-requested-placeholder.json` — explicitly blocked exact-target shape.
* `zcode-project-config-registered-example.json` — nearest registered example, template only.
* `diagnostic-prompt.txt` — bounded synthetic prompt.
* `zcode-native-cua-template.sh` — conditional desktop-CUA command; no launch performed.
* `headless-browser-variant.txt` — separate managed-CDP note.
