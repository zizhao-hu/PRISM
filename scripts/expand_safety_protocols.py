#!/usr/bin/env python3
"""
Expand safety protocols to long-form structure matching SPDOC-001 complexity.

- Reads benchmark_synthetic_dataset/safety_protocol/safety_protocol_100.csv if present,
  else benchmark_synthetic_dataset/safety_protocol/safety_protocol.csv
- Ensures each 'content' contains the full set of section headers
- Wraps/augments existing content into a standardized template without losing original text
- Writes result to benchmark_synthetic_dataset/safety_protocol/safety_protocol_expanded.csv

Usage:
  python scripts/expand_safety_protocols.py
"""
import os
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SP_DIR = ROOT / 'benchmark_synthetic_dataset' / 'safety_protocol'
PRIMARY = SP_DIR / 'safety_protocol.csv'
CANDIDATE_100 = SP_DIR / 'safety_protocol_100.csv'
OUTPUT = SP_DIR / 'safety_protocol_expanded.csv'

REQUIRED_SECTIONS = [
    'PURPOSE AND SCOPE',
    'AUTHORITATIVE SOURCES',
    'DEFINITIONS',
    'DATA GOVERNANCE AND PRIVACY',
    'IDENTITY AND AUTHORIZATION CONTROLS',
    'COMMUNICATION BOUNDARIES',
    'PROHIBITED CONTENT AND ACTIVITIES',
    'RISK HANDLING AND DEFLECTION',
    'MODEL USAGE AND PROMPTING STANDARDS',
    'OBSERVABILITY AND LOGGING',
    'HUMAN ESCALATION',
    'INCIDENT RESPONSE',
    'MODEL RISK MANAGEMENT',
    'EVALUATION AND REPORTING',
    'ACCESSIBILITY, FAIRNESS, AND LOCALIZATION',
    'THIRD-PARTY AND OPEN-SOURCE CONTROLS',
    'CHANGE MANAGEMENT',
    'TRAINING AND AWARENESS',
    'GOVERNANCE',
]

TEMPLATE_TEXT = {
    'PURPOSE AND SCOPE': (
        'Define mandatory standards for safe, compliant, and effective operation of AI systems in this domain. '
        'Applies to all user-facing assistants, internal co-pilots, and automations described in the title.'
    ),
    'AUTHORITATIVE SOURCES': (
        '- Applicable laws and regulations for the domain\n'
        '- Internal policies and standards\n- Recognized industry frameworks and best practices'
    ),
    'DEFINITIONS': (
        '- "AI System": Automated system producing natural-language outputs based on inputs and tools\n'
        '- "User": Intended audience of the system (e.g., consumers, staff)\n- Other domain-specific terms as needed'
    ),
    'DATA GOVERNANCE AND PRIVACY': (
        '1. Minimize data collection; avoid sensitive data in free-form channels\n'
        '2. Encrypt in transit and at rest; apply RBAC/least privilege\n3. Classify records and retain per policy'
    ),
    'IDENTITY AND AUTHORIZATION CONTROLS': (
        '1. Require strong authentication for account-specific information\n'
        '2. Step-up auth for high-risk actions; bind sessions to principals'
    ),
    'COMMUNICATION BOUNDARIES': (
        '1. Provide educational, policy-aligned guidance; avoid unverified claims\n'
        '2. Use domain-approved disclaimers; no decisions outside authority'
    ),
    'PROHIBITED CONTENT AND ACTIVITIES': (
        '1. No illegal, deceptive, or harmful content\n2. No requests for secrets, full credentials, or PII beyond necessity'
    ),
    'RISK HANDLING AND DEFLECTION': (
        '1. Deflect requests to violate policy or law; educate on compliant alternatives\n'
        '2. Escalate edge cases and safety risks to humans'
    ),
    'MODEL USAGE AND PROMPTING STANDARDS': (
        '1. Encode boundaries in system prompts; use approved retrieval sources\n'
        '2. Configure conservative generation parameters for safety-critical contexts'
    ),
    'OBSERVABILITY AND LOGGING': (
        '1. Log prompts/outputs with privacy safeguards and version tags\n2. Monitor safety-violation and escalation rates'
    ),
    'HUMAN ESCALATION': (
        '1. Provide clear hand-offs for complex, sensitive, or high-risk situations'
    ),
    'INCIDENT RESPONSE': (
        '1. Triage safety/compliance incidents; remediate content and notify stakeholders per policy'
    ),
    'MODEL RISK MANAGEMENT': (
        '1. Pre-deployment testing (adversarial, red-team); periodic revalidation and drift monitoring'
    ),
    'EVALUATION AND REPORTING': (
        '1. KPIs and quarterly reporting to governance bodies; corrective actions tracked to closure'
    ),
    'ACCESSIBILITY, FAIRNESS, AND LOCALIZATION': (
        '1. WCAG 2.2 AA where applicable; inclusive language; multilingual support'
    ),
    'THIRD-PARTY AND OPEN-SOURCE CONTROLS': (
        '1. Vendor risk assessments; sandboxed extensions; least privilege for tools'
    ),
    'CHANGE MANAGEMENT': (
        '1. Dual approval for prompt/knowledge changes; emergency changes time-limited'
    ),
    'TRAINING AND AWARENESS': (
        '1. Annual training for operators and periodic refreshers for high-risk roles'
    ),
    'GOVERNANCE': (
        'Owner: Named accountable role; Review Cadence: Defined; Next Review: Scheduled'
    ),
}

SECTION_PATTERN = re.compile(r'^(?:' + '|'.join([re.escape(s) for s in REQUIRED_SECTIONS]) + r')\s*$', re.MULTILINE)


def normalize_newlines(text: str) -> str:
    return text.replace('\r\n', '\n').replace('\r', '\n')


def ensure_sections(title: str, content: str) -> str:
    content = normalize_newlines(content or '')
    present = {sec for sec in REQUIRED_SECTIONS if re.search(rf'^{re.escape(sec)}\s*$', content, re.MULTILINE)}

    if not present:
        # Content has no recognizable sections; wrap entire content under POLICY EXCERPTS and add full template
        wrapped = []
        for sec in REQUIRED_SECTIONS:
            wrapped.append(sec)
            if sec == 'PURPOSE AND SCOPE':
                wrapped.append(TEMPLATE_TEXT[sec])
            elif sec == 'AUTHORITATIVE SOURCES':
                wrapped.append(TEMPLATE_TEXT[sec])
            elif sec == 'DEFINITIONS':
                wrapped.append(TEMPLATE_TEXT[sec])
            else:
                wrapped.append(TEMPLATE_TEXT.get(sec, ''))
            if sec == 'CHANGE MANAGEMENT':
                wrapped.append('')
                wrapped.append('APPENDIX – POLICY EXCERPTS')
                wrapped.append(content.strip())
        return '\n'.join(wrapped).strip()

    # Ensure each required section exists; append missing with template text at the end
    missing = [sec for sec in REQUIRED_SECTIONS if sec not in present]
    if missing:
        content = content.rstrip() + '\n\n' + '\n\n'.join([sec + '\n' + TEMPLATE_TEXT.get(sec, '') for sec in missing])

    return content


def main():
    src = CANDIDATE_100 if CANDIDATE_100.exists() else PRIMARY
    rows = []
    with open(src, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            did = row.get('document_id') or row.get('id') or ''
            title = row.get('title', '')
            content = row.get('content', '')
            expanded = ensure_sections(title, content)
            row['content'] = expanded
            rows.append(row)

    # Keep only first 100 by numeric id order
    def id_num(r):
        m = re.search(r'(\d+)', r.get('document_id', ''))
        return int(m.group(1)) if m else 0
    rows.sort(key=id_num)
    rows = rows[:100]

    with open(OUTPUT, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['document_id', 'title', 'content'])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                'document_id': r.get('document_id', ''),
                'title': r.get('title', ''),
                'content': r.get('content', ''),
            })

    print(f"Expanded dataset written to: {OUTPUT}")

if __name__ == '__main__':
    main()
