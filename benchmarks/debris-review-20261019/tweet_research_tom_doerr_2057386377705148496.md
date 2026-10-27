# Research brief: tweet by Tom Dörr — github.com/seo-skills/seo-audit-skill
Task date: 2026-10-15 · Sources: cdn.syndication.twimg.com JSON, api.fxtwitter.com JSON, GitHub API + shallow clone of the repo, raw README/SKILL.md, tweet PNG (inspected directly)

## 1. The tweet
- URL: https://x.com/tom_doerr/status/2057386377705148496
- Verbatim text (display range 0–76): `Audits websites with 251 rules across 20 categories` + a link t.co/sZRIEDLDo0 → https://github.com/seo-skills/seo-audit-skill and one photo.
- Author: Tom Dörr (@tom_doerr), blue-verified, ~222.0K followers; bio: "Follow for posts about GitHub repos, DSPy, and agents".
- Posted: 2026-05-21T09:01:51Z. Media: 1 PNG photo (678×754) — a screenshot of the repo's README page ("SEOmator Audit" heading, features list).
- Engagement (fx mirror, corroborated by official syndication JSON for likes/replies): 229 likes · 18 reposts · 4 replies · 39,266 views. Bookmark count not exposed. Edit history: none (isEdited: false).

## 2. Linked artifact
- Repo: github.com/seo-skills/seo-audit-skill — npm package `@seomator/seo-audit` (bin `seomator`), v6.0.0, MIT, TypeScript.
- GitHub: 551 stars · 70 forks · created 2026-01-23 · pushed 2026-10-15 (today); 19 npm releases since Jan 2026, latest published 2026-10-15.
- Rule-count drift over time: tweet (May) = 251 · GitHub repo description (current) = 332 · README on main (current) = 373. Source grep finds 377 rule-id strings. Counts evolve weekly; README claims 373 across 20 categories.
- UNVERIFIED: whether Tom Dörr is affiliated with SEOmater/seomator (repo org is "seo-skills", author metadata is "seomator"). His bio reads as curator; the tweet is promotional-style but affiliation not confirmable from fetched sources.

## 3. What it does technically (grounded in cloned source)
- Self-contained crawler/auditor: fetches pages itself (cheerio for static HTML; Playwright for Core Web Vitals LCP/CLS/FCP/TTFB/INP and raw-vs-rendered DOM analysis for SPA/SSR gaps).
- 20 categories incl. Core (24), Performance (28), Links (27), Security (26), Crawlability (38), Structured Data (19), JavaScript Rendering (16), Accessibility (36), Content (27), E-E-A-T (16), Mobile, i18n, HTML validation, redirects, legal.
- AI/GEO Readiness category: 13 rules, exact IDs in src/rules/geo/: geo-semantic-html, geo-content-structure, geo-ai-bot-access (citation crawlers allowed vs GPTBot-type training crawlers reported), geo-llms-txt, geo-schema-drift (JSON-LD vs visible content), geo-content-signals, geo-noai-signals, geo-agents-md, geo-well-known (MCP/agent-card manifest), geo-rsl-license, geo-markdown-response, geo-markdown-page, geo-pay-per-crawl (HTTP 402 payment terms).
- Commands: audit <url>, serve (local API on 127.0.0.1:7360 with printed token), init, crawl, analyze, report, compare (diffs two audits by rule ID — "what the deploy changed"), config, db, self doctor. Concurrent crawling, SQLite (better-sqlite3) storage with audit history.
- Output: 5 formats — console, JSON, HTML, Markdown, "LLM-optimized" XML. Score 0–100 with A–F grade, per-category pass/warn/fail, severity, affected URLs, fix suggestions; exit codes + GitHub Actions/GitLab CI examples; ships a Claude Code skill (SKILL.md, `allowed-tools: Bash(seomator:*)`).
- No API keys, no paid API, no telemetry endpoints found in src (grep of external URLs in *.ts: only test domains, localhost, and doc links). Playwright just drives a local browser.

## 4. Fit vs our geo-audit-v3.py / pipeline.py
- Strongest slot — scoring/rules: the 13 AI/GEO Readiness rules overlap almost 1:1 with our AEO visibility checks (llms.txt, ai-bot access, schema drift, markdown representations, .well-known). Use as an independent second scorer to validate our rule logic, and crib the rule list to expand coverage (e.g. RSL license, 402 pay-per-crawl are checks we likely lack).
- E-E-A-T (16 rules) also maps to citation-worthiness scoring in our reports.
- Reporting/data: JSON output maps cleanly into our findings CSV; `compare` diff-by-rule between two runs fits our dated-report idea ("what changed since last audit") almost off-the-shelf.
- Not a fit: it does NOT query LLM answer engines or SERPs — no AI-answer capture. It audits on-site technical readiness only, so it's complementary to (not a replacement for) our answer-engine visibility measurement.
- Effort: low–moderate. Node 20.3+; `npm i -g @seomator/seo-audit`; optional Playwright browser (~150 MB) only if we want CWV/rendering rules. Integration = subprocess call + JSON parse in pipeline.py; days not weeks.
- Licensing/keys: MIT (LICENSE + README badge + SKILL.md frontmatter). No API keys or paid service; the seomator.com free-audit-tool homepage is just a marketing link to the vendor's web UI. Caution: maintained by commercial vendor SEOmator — pin the npm version (supply-chain), and rule count/weights change frequently (251→373 in 5 months), so pin and diff upstream rule definitions.

## Executive summary (5)
- Tweet (2026-05-21, @tom_doerr): "Audits websites with 251 rules across 20 categories", one link + one screenshot; 229 likes / 18 reposts / 4 replies / 39.3K views.
- Links to github.com/seo-skills/seo-audit-skill — `@seomator/seo-audit`: MIT-licensed TypeScript CLI + Electron app + Claude skill, actively maintained (v6.0.0 today), self-hosted, no API keys.
- A Playwright+cheerio crawler scoring sites across 20 categories / 373 rules (251 at tweet time) with 0–100 score, JSON/MD/HTML/XML outputs and audit-diff command.
- Its AI/GEO Readiness (13 rules) and E-E-A-T (16 rules) categories are the highest-value slot-ins for our pipeline: independent validation of our AEO scoring plus new checks (llms.txt, bot access, RSL, pay-per-crawl); JSON output drops into our findings CSV.
- Not an answer-engine monitor (no SERP/LLM capture); integration is a subprocess + JSON parse (days), MIT so licensing is clean — pin the version since rule counts churn.