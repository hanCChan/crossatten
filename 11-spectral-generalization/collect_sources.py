"""Save publisher-deposited metadata, including access failures, without technical inference."""
import json
import os
import urllib.request
from datetime import datetime, timezone
from core import HERE, save


def main():
    rows=[]
    for doi in ("10.1109/LCOMM.2026.3718894","10.2139/ssrn.7389346"):
        url="https://api.crossref.org/works/"+doi
        row=dict(doi=doi,url=url,evidence_level="publisher-deposited metadata only")
        proxies=[os.environ.get("HTTPS_PROXY"),"http://star-proxy.oa.com:3128"]
        errors=[]
        for proxy in dict.fromkeys(proxies):
            if not proxy:
                continue
            try:
                opener=urllib.request.build_opener(
                    urllib.request.ProxyHandler({"http":proxy,"https":proxy}))
                with opener.open(url,timeout=40) as response:
                    message=json.load(response)["message"]
                row["metadata"]={k:message[k] for k in (
                    "DOI","title","author","publisher","container-title","type",
                    "published","published-online","created","URL") if k in message}
                row["accessed"]=True
                break
            except Exception as exc:
                errors.append(type(exc).__name__)
        row["access_errors"]=errors
        rows.append(row)
    save(HERE/"sources.json",dict(accessed_utc=datetime.now(timezone.utc).isoformat(),
        scope="two known DOIs; targeted metadata check, not exhaustive novelty search",
        sources=rows,technical_comparison_from_metadata=False))
    print([(r["doi"],r.get("accessed",False)) for r in rows])


if __name__=="__main__":
    main()
