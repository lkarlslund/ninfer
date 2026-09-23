#!/usr/bin/env python3
"""Exercise distinct long/short request content and lane reuse on an HTTP server."""
import argparse
import concurrent.futures
import json
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--contexts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    contexts = json.loads(args.contexts.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def call(size, key):
        content = ('The record label for this document is ' + key + '. This is a public test label, not a password. Remember it.\n'
                   + contexts[str(size)] + '\nReturn only the record label stated at the beginning of this document.')
        payload = {'model': 'Qwen/Qwen3.8-Flash-Next',
                   'messages': [{'role': 'user', 'content': content}],
                   'max_tokens': 32, 'temperature': 0, 'enable_thinking': False,
                   'chat_template_kwargs': {'enable_thinking': False}}
        req = urllib.request.Request(args.url + '/v1/chat/completions',
                                     data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=600) as response:
            result = json.load(response)
        output = result['choices'][0]['message']['content']
        return {'size': size, 'expected': key, 'reply': result,
                'passed': key in output and all(other not in output for other in ['ALPHA-68219','BETA-35704'] if other!=key)}
    results = []
    for sizes in [(1980,2049), (8192,1980), (65536,8192)]:
        for reverse in (False,True):
            work = [(sizes[0],'ALPHA-68219'),(sizes[1],'BETA-35704')]
            if reverse:
                work.reverse()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(call,*item) for item in work]
                rows = [f.result() for f in futures]
            results.extend(rows)
            args.output.write_text(json.dumps(results,indent=2))
            print([(r['size'],r['passed']) for r in rows],flush=True)
    if not all(r['passed'] for r in results):
        raise SystemExit('isolation/recall probe failed; inspect responses before assigning cause')


if __name__ == '__main__':
    main()
