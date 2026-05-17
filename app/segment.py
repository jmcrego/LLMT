from typing import List, Dict, Tuple


# -----------------------------
# Step 1: extract punctuation anchors
# -----------------------------
def extract_anchors(text: str) -> List[int]:
    """
    Returns character indices of valid segmentation boundaries.
    Only punctuation-based anchors (NO fallback logic).
    """

    anchors = []

    for i, ch in enumerate(text):
        if ch in [".", "!", "?", ";", ",", "\n"]:
            anchors.append(i + 1)  # cut AFTER punctuation

    # ensure start and end anchors exist
    if 0 not in anchors:
        anchors = [0] + anchors
    if len(text) not in anchors:
        anchors.append(len(text))

    return anchors


# -----------------------------
# Cache system
# -----------------------------
class SegmentCache:
    def __init__(self):
        self.cache = set()

    def has(self, segment: str) -> bool:
        return segment in self.cache

    def add(self, segment: str):
        self.cache.add(segment)


# -----------------------------
# Scoring function
# -----------------------------
def score(segment: str, cache: SegmentCache) -> int:

    s = segment.strip()

    # cache reward
    cache_reward = 10 if cache.has(s) else 0

    # punctuation strength
    if s.endswith((".", "!", "?")):
        punct = 3
    elif s.endswith(";"):
        punct = 2
    elif s.endswith(","):
        punct = 1
    else:
        punct = 0

    return cache_reward + punct


# -----------------------------
# Viterbi over anchor graph
# -----------------------------
def viterbi_anchor_segmentation(text: str, cache: SegmentCache) -> List[str]:

    anchors = extract_anchors(text)
    n = len(anchors)

    # DP arrays
    dp = [-float("inf")] * n
    back = [-1] * n

    dp[0] = 0

    # -----------------------------
    # DP over anchor indices only
    # -----------------------------
    for j in range(1, n):

        for i in range(0, j):

            segment = text[anchors[i]:anchors[j]]

            s = dp[i] + score(segment, cache)

            if s > dp[j]:
                dp[j] = s
                back[j] = i

    # -----------------------------
    # reconstruct best path
    # -----------------------------
    segments = []
    j = n - 1

    while j > 0:
        i = back[j]
        if i == -1:
            break

        seg = text[anchors[i]:anchors[j]].strip()
        segments.append(seg)
        j = i

    segments.reverse()

    # update cache
    for seg in segments:
        cache.add(seg)

    return segments


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":

    text = (
        "Yesterday I bought a car for one million euros. "
        "It was built in the 1950s, and it had many owners; "
        "it was also restored several times! "
        "Finally it burned in an accident."
    )

    cache = SegmentCache()

    result = viterbi_anchor_segmentation(text, cache)

    for r in result:
        print("SEG:", r)