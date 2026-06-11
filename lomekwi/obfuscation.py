"""Obfuscation token machinery.

The rigor is NOT in hand-picking "neutral-looking" tokens. Hand-picked tokens
carry baggage: 'frob' = frobnicate (a tool/manipulate connotation), 'silex' =
Latin for flint, 'velax' reads as a brand, 'korvan' as a proper noun. So we do
NOT fix "byproduct = frob".

What makes the method sound is the relabeling protocol (formalism sec. 4):
  (1) per episode, draw the label->element assignment UNIFORMLY AT RANDOM from a
      large pool; (2) average results over many such relabelings.
Under random assignment the label is independent of the element's structural
role, so by Lemma 1 any residual token baggage is uninformative IN EXPECTATION.
This holds for ANY pool, however messy individual tokens are (even subword
leakage from BPE is randomly assigned to roles, hence uninformative on average).

A generated, dictionary-filtered, pronounceable pool is a SECOND-ORDER nicety:
it lowers per-episode noise and makes single transcripts cleaner, so fewer
relabelings are needed. It is not a substitute for the randomization.

Honest limitation: we cannot fully control tokenization across providers
(Anthropic's tokenizer is not public), so exact exchangeability holds only in
expectation over relabelings -- which is exactly what (1)+(2) deliver.
"""

from __future__ import annotations

import functools
import random
import string
from pathlib import Path

# Obfuscation scheme selector. "tokens" = pronounceable nonsense tokens (the
# default, documented above). "letter" = a single random letter per element
# (a maximally minimal obfuscation; same randomization protocol, smaller pool).
# Callers may override per-call via assign(..., scheme=...); this global is the
# fallback so it can be flipped from config (see scripts/sweep_config.py).
DEFAULT_SCHEME = "tokens"

# Phonotactics: pronounceable, easy to read and copy back exactly.
ONSETS = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z",
          "br", "dr", "fl", "gl", "gr", "kr", "pl", "pr", "sk", "sl", "sn",
          "sp", "st", "tr", "vr", "thr", "shp"]
VOWELS = ["a", "e", "i", "o", "u"]
CODAS = ["", "", "b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "x",
         "ld", "lk", "lt", "mp", "nd", "nk", "nt", "rd", "rk", "rn", "sk", "st"]

# Confusable / ugly fragments to skip outright.
_BAN_SUBSTR = ("rn", "cl", "vv", "ii", "uu")

_DICT_PATH = Path("/usr/share/dict/words")


@functools.lru_cache(maxsize=1)
def _english() -> frozenset[str]:
    if not _DICT_PATH.exists():
        return frozenset()
    words = set()
    for w in _DICT_PATH.read_text(errors="ignore").split():
        words.add(w.strip().lower())
    return frozenset(words)


def _syllable(rng: random.Random) -> str:
    return rng.choice(ONSETS) + rng.choice(VOWELS) + rng.choice(CODAS)


def _candidate(rng: random.Random) -> str:
    n = rng.choice([1, 2, 2])  # bias toward 2 syllables
    tok = "".join(_syllable(rng) for _ in range(n))
    return tok


def _ok(tok: str) -> bool:
    if not (4 <= len(tok) <= 8):
        return False
    if any(b in tok for b in _BAN_SUBSTR):
        return False
    if tok in _english():
        return False
    # reject if a real word is a long prefix/suffix (sub-word leakage heuristic)
    eng = _english()
    if eng and (tok[:4] in eng or tok[-4:] in eng):
        return False
    return True


def _lev(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def generate_pool(size: int = 200, seed: int = 0) -> list[str]:
    """A large pool of dictionary-filtered pronounceable nonsense tokens."""
    rng = random.Random(seed)
    pool: list[str] = []
    seen: set[str] = set()
    tries = 0
    while len(pool) < size and tries < size * 200:
        tries += 1
        tok = _candidate(rng)
        if tok in seen or not _ok(tok):
            continue
        seen.add(tok)
        pool.append(tok)
    return pool


def assign_letters(elements: list[str], seed: int) -> dict[str, str]:
    """Single-random-letter obfuscation: draw a distinct lowercase letter for
    each latent element. Same per-episode uniform-random relabeling protocol as
    assign() (sec. 4), just with the 26-letter alphabet as the pool. Distinct
    letters trivially satisfy any min_dist >= 1, so confusability is a non-issue
    here. Limited to 26 elements by construction."""
    if len(elements) > len(string.ascii_lowercase):
        raise ValueError(f"single-letter scheme supports <= 26 elements, got "
                         f"{len(elements)}; use scheme='tokens'")
    rng = random.Random(seed * 7919 + 13)
    letters = list(string.ascii_lowercase)
    rng.shuffle(letters)
    return dict(zip(elements, letters))


# Alphabet for the "alnum" scheme: [a-zA-Z0-9].
_ALNUM = string.ascii_letters + string.digits


def assign_alnum(elements: list[str], seed: int, min_len: int = 4,
                 max_len: int = 8, min_dist: int = 3) -> dict[str, str]:
    """Random-alphanumeric obfuscation: a distinct random [a-zA-Z0-9] string of
    length in [min_len, max_len] per element. Same per-episode uniform-random
    relabeling protocol as assign() (sec. 4); pairwise edit distance >= min_dist
    so labels are never confusable/mistypable."""
    rng = random.Random(seed * 7919 + 13)
    chosen: list[str] = []
    tries = 0
    cap = len(elements) * 5000
    while len(chosen) < len(elements) and tries < cap:
        tries += 1
        n = rng.randint(min_len, max_len)
        tok = "".join(rng.choice(_ALNUM) for _ in range(n))
        if all(_lev(tok, c) >= min_dist for c in chosen):
            chosen.append(tok)
    if len(chosen) < len(elements):
        raise ValueError(f"could not draw {len(elements)} alnum labels at "
                         f"min_dist={min_dist}; lower min_dist or widen length")
    return dict(zip(elements, chosen))


def assign(elements: list[str], seed: int, min_dist: int = 3,
           pool_size: int = 300, scheme: str | None = None) -> dict[str, str]:
    """Draw a uniformly-random label for each latent element, with pairwise edit
    distance >= min_dist so the model never confuses/mistypes two tokens.

    `seed` indexes the relabeling: the SAME structure run under different seeds
    gives independent labelings (this is the Sym(V) sampling of formalism sec. 4).

    `scheme` selects the label pool: "tokens" (pronounceable nonsense, default),
    "letter" (a single random letter each), or "alnum" (a random alphanumeric
    string of length 4-8 each). None falls back to DEFAULT_SCHEME.
    """
    scheme = scheme or DEFAULT_SCHEME
    if scheme == "letter":
        return assign_letters(elements, seed)
    if scheme == "alnum":
        return assign_alnum(elements, seed, min_dist=min_dist)
    if scheme != "tokens":
        raise ValueError(f"unknown obfuscation scheme {scheme!r}")
    rng = random.Random(seed * 7919 + 13)
    # Build the pool with a seed-independent base so pools are comparable, then
    # shuffle order per relabeling so the *assignment* is what randomizes.
    pool = generate_pool(size=pool_size, seed=0)
    rng.shuffle(pool)
    chosen: list[str] = []
    for tok in pool:
        if all(_lev(tok, c) >= min_dist for c in chosen):
            chosen.append(tok)
        if len(chosen) == len(elements):
            break
    if len(chosen) < len(elements):
        raise ValueError(f"pool too small for {len(elements)} elements at "
                         f"min_dist={min_dist}; raise pool_size or lower min_dist")
    return dict(zip(elements, chosen))


if __name__ == "__main__":
    pool = generate_pool(40, seed=0)
    print(f"sample pool (40): {pool}\n")
    # demo: relabel the tool world's latent elements over 3 relabelings
    elements = ["door", "key", "byproduct", "machine"]
    for s in range(3):
        print(f"relabeling seed={s}: {assign(elements, seed=s)}")
