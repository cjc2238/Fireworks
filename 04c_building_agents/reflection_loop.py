"""Evaluator-optimizer loop with a REAL verifier: the model writes SQL, we execute it against
SQLite, and feed back errors or result mismatches until the query is correct.
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sqlite3

from fwlearn import MODEL, banner, client

db = sqlite3.connect(":memory:")
db.executescript("""
CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT, region TEXT);
CREATE TABLE usage(customer_id INTEGER, month TEXT, tokens INTEGER);
INSERT INTO customers VALUES (1,'acme','EUROPE'),(2,'zeta','US'),(3,'nova','EUROPE'),(4,'kilo','APAC');
INSERT INTO usage VALUES (1,'2026-08',900),(1,'2026-09',1200),(2,'2026-09',300),(3,'2026-08',50),
                         (3,'2026-09',700),(4,'2026-09',1500),(2,'2026-08',400);
""")
SCHEMA = "customers(id, name, region); usage(customer_id, month 'YYYY-MM', tokens)"
QUESTION = "For each region, total September 2026 tokens, highest first. Columns: region, total."
EXPECTED = db.execute("""SELECT c.region, SUM(u.tokens) total FROM usage u JOIN customers c ON c.id=u.customer_id
                         WHERE u.month='2026-09' GROUP BY c.region ORDER BY total DESC""").fetchall()


def verify(sql: str) -> tuple[bool, str]:
    """The evaluator. Deterministic, not an LLM opinion."""
    try:
        rows = db.execute(sql).fetchall()
    except sqlite3.Error as e:
        return False, f"SQL error: {e}"
    if rows != EXPECTED:
        return False, f"Query ran but returned {rows[:5]}; expected {len(EXPECTED)} rows sorted by total desc."
    return True, "correct"


fw = client()
messages = [
    {"role": "system", "content": f"Write one SQLite query. Schema: {SCHEMA}. Reply with SQL only, no fences."},
    # The deliberately vague first attempt tends to miss the month filter or the join.
    {"role": "user", "content": "Total tokens by region, biggest first."},
]
for it in range(1, 5):
    r = fw.chat.completions.create(model=MODEL, messages=messages, temperature=0, reasoning_effort="none")
    sql = r.choices[0].message.content.strip().removeprefix("```sql").removesuffix("```").strip()
    ok, feedback = verify(sql)
    banner(f"iteration {it}: {'PASS' if ok else 'FAIL'}")
    print(sql, "\n->", feedback)
    if ok:
        break
    messages += [{"role": "assistant", "content": sql},
                 {"role": "user", "content": f"Verifier feedback: {feedback}\nRequirement: {QUESTION}\nFix the query."}]
print("\nexpected:", EXPECTED)
