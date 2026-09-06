#!/usr/bin/env python3
"""Three synthetic R requests through Kaggle's model proxy; no benchmark publication.

Run with the authenticated Kaggle CLI's Python environment. Credentials remain
in memory. A journal is created before network use and refuses a repeat run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

CASES = [
    ('missing', 'Return only one R expression that counts missing values in x.', 'sum(is.na(x))'),
    ('rename', 'Rename only argument x to values in this R function. Return code only: f <- function(x) { sum(x) }', 'f <- function(values) { sum(values) }'),
    ('noop', 'This R code already counts missing values correctly. Return it unchanged, no explanation: sum(is.na(x))', 'sum(is.na(x))'),
]


def failure_code(error):
    if isinstance(error, urllib.error.HTTPError): return error.code
    # Kaggle SDK translates these responses into ValueError and removes the
    # response object. Recognize its fixed public messages only.
    if isinstance(error, ValueError):
        if str(error).startswith('Authentication failed (403)'): return 403
        if str(error).startswith('Endpoint not found (404)'): return 404
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    args = parser.parse_args()
    root = args.state.expanduser().resolve()
    from kaggle_job import ensure_external_state, save
    ensure_external_state(root)
    root.mkdir(parents=True, exist_ok=False)
    report = {'kind': 'benchmark-proxy-capability', 'max_requests': 3,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'cases': CASES, 'results': [], 'status': 'authenticating'}
    save(root, report)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        config = api._fetch_model_proxy_env(source='auth')
        base = config['MODEL_PROXY_URL'].rstrip('/')
        parsed = urllib.parse.urlparse(base)
        if parsed.scheme != 'https' or not parsed.hostname.endswith('.kaggle.net'):
            raise ValueError('Unexpected model-proxy origin')
        def call(path, payload=None):
            data = json.dumps(payload).encode() if payload else None
            request = urllib.request.Request(base + path, data=data, headers={
                'Authorization': 'Bearer ' + config['MODEL_PROXY_API_KEY'],
                'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        listing = call('/models')
        models = [m['id'] for m in listing['data']]
        report['available_models'] = models
        chosen = next((m for m in models if 'gemini-3.1-flash-lite' in m), None)
        if chosen is None: chosen = next((m for m in models if 'gemini' in m and 'flash' in m), None)
        if chosen is None: raise ValueError('No intended small model available; no substitute billed')
        report.update(model=chosen, status='ready')
        save(root, report)
        for name, prompt, expected in CASES:
            # Persist before each request. Lost replies consume this attempt.
            report['status'] = 'requesting:' + name
            save(root, report)
            start = time.monotonic()
            response = call('/chat/completions', dict(model=chosen,
                messages=[dict(role='user', content=prompt)], max_tokens=128, temperature=0))
            content = response['choices'][0]['message']['content']
            report['results'].append(dict(case=name, text=content, expected=expected,
                exact_match=content.strip() == expected, usage=response.get('usage'),
                response_id=response.get('id'), elapsed_seconds=time.monotonic()-start))
            save(root, report)
        report['status'] = 'completed'
    except Exception as error:
        # SDK messages can contain connection details. Record the class and HTTP
        # code, not a credential-bearing request representation or response body.
        report.update(status='blocked', error_type=type(error).__name__,
                      http_status=failure_code(error))
    save(root, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
