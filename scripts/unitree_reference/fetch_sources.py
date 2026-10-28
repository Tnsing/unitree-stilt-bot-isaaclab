#!/usr/bin/env python3
"""Download only locked upstream source and G1 assets; verify every payload hash.

Normal HTTPS_PROXY is supported. Does not install or modify system environments.
"""
import concurrent.futures,hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def fetch(item):
    dest,url,expected=item
    if dest.is_file() and hashlib.sha256(dest.read_bytes()).hexdigest()==expected:return 'cached '+str(dest.relative_to(ROOT))
    for attempt in range(3):
        try:
            data=urllib.request.urlopen(url,timeout=20).read()
            if hashlib.sha256(data).hexdigest()!=expected:raise ValueError(f'Hash mismatch: {url}')
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);return 'downloaded '+str(dest.relative_to(ROOT))
        except Exception:
            if attempt==2:raise

def main():
    source=json.loads((ROOT/'configs/unitree_upstream_lock.json').read_text());asset=json.loads((ROOT/'configs/unitree_asset_lock.json').read_text())
    items=[]
    for rel,digest in source['files'].items():
        items.append((ROOT/'third_party/unitree_rl_lab'/rel,f"https://raw.githubusercontent.com/{source['repo']}/{source['commit']}/{rel}",digest))
    for rel,digest in asset['files'].items():
        items.append((ROOT/'assets/unitree_ros/g1_description'/rel,f"https://raw.githubusercontent.com/{asset['source']}/{asset['ref']}/robots/g1_description/{rel}",digest))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for result in pool.map(fetch,items):print(result,flush=True)
    (ROOT/'third_party/unitree_rl_lab/UPSTREAM.json').write_text(json.dumps(source,indent=2))
    (ROOT/'assets/unitree_ros/g1_description/PROVENANCE.json').write_text(json.dumps(asset,indent=2))
if __name__=='__main__':main()
