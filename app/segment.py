from typing import List
import bisect
import re
import time


# =========================================================
# CACHE
# =========================================================
class SegmentCache:
    def __init__(self):
        self.cache = set()

    def has(self, segment: str) -> bool:
        return segment.strip() in self.cache

    def add(self, segment: str):
        self.cache.add(segment.strip())


# =========================================================
# ANCHOR EXTRACTION
# =========================================================
def extract_anchors(text: str, patterns: dict, max_len: int) -> List[int]:
    anchor_set = {0, len(text)}
    for pat in patterns:
        for m in pat.finditer(text):
            anchor_set.add(m.start() + 1)

    # while there be sequences without anchors that exceed the max_len, 
    # we add intermediate anchors every ~max_len chars (on space chars)
    # this ensures that the algorithm can break long segments that would otherwise exceed the model context window
    sorted_anchors = sorted(anchor_set)
    i = 0
    while i < len(sorted_anchors) - 1:
        start = sorted_anchors[i]
        end = sorted_anchors[i + 1]

        if end - start > max_len:
            # find the last space char before start + max_len
            anchor_set.add(text.rfind(" ", start, start + max_len) + 1)

        i += 1
        sorted_anchors = sorted(anchor_set)

    return sorted(anchor_set)


# =========================================================
# EDGE SCORING
# =========================================================
def score_edge(text: str,
               i: int,
               j: int,
               patterns: dict,
               cache: SegmentCache,
               max_len: int,
               max_len_penalty: int,
               cache_reward: int) -> int:

    score = 0

    segment = text[i:j]

    if cache.has(segment):
        score += cache_reward

    context = text[j - 1:j + 4]
    matched = [val for pat, val in patterns.items() if pat.match(context)]
    # if multiple patterns match, the one with the scores are summed
    if matched:
        score += sum(matched)

    if j-i > max_len:
        score -= max_len_penalty

    return score


# =========================================================
# VITERBI SEGMENTATION
# =========================================================
def viterbi_segment(text: str,
                    cache: SegmentCache,
                    max_len: int,
                    max_len_penalty: int,
                    cache_reward: int,
                    edge_penalty: int,
                    patterns: dict):

    anchors = extract_anchors(text, patterns, max_len)
    print(f"Anchors: {anchors}")
    n = len(anchors)

    dp   = [-float("inf")] * n
    back = [-1] * n

    dp[0] = 0

    # ------------------------------------------------
    # DP over anchor graph
    # ------------------------------------------------
    for j in range(1, n):

        for i in range(j):

            i_pos = anchors[i]
            j_pos = anchors[j]

            if dp[i] == -float("inf"):
                continue

            s = (
                dp[i]
                + score_edge(text, i_pos, j_pos, patterns, cache, max_len, max_len_penalty, cache_reward)
                - edge_penalty
            )

            if s > dp[j]:
                dp[j] = s
                back[j] = i

    # ------------------------------------------------
    # reconstruction
    # ------------------------------------------------
    segments = []
    j = n - 1

    while j > 0:
        i = back[j]
        segments.append({"start": anchors[i], "end": anchors[j], "score": dp[j] - dp[i]})
        j = i

    segments.reverse()

    # ------------------------------------------------
    # cache update
    # ------------------------------------------------
    for seg in segments:
        cache.add(segment=text[seg["start"]:seg["end"]])

    return anchors, segments


# =========================================================
# EXAMPLE
# =========================================================
if __name__ == "__main__":

    text = """Yesterday I bought a very beautiful and exceptionally rare book for around one million euros, it had originally been created in the 1950s and had previously belonged to several famous cinema directors, painters, collectors, and aristocrats living throughout the 1960s, 1970s, and 1980s, after having been exhibited in museums, and featured repeatedly in magazines, admired by enthusiasts from all around the world. It has been sold several times and suffered renovations very often during the 1990s after it had an accident in which it burned. Today I replaced it with a car."""
    #text = """Yesterday I bought a very beautiful and exceptionally rare book for around one million euros, it had originally been created in the 1950s and had previously belonged to several famous cinema directors, painters, collectors, and aristocrats living throughout the 1960s, 1970s, and 1980s, after having been exhibited in museums, and featured repeatedly in magazines, admired by enthusiasts from all around the world in the day of heaven and over the rainbows. It has been sold several times and suffered renovations very often during the 1990s after it had an accident in which it burned. Today I replaced it with a car."""
    # text = """The line is crackly. But the voice of Mehrab Abdollahzadeh is clear and, given the circumstances, surprisingly steady. He's on death row in western Iran. He speaks quickly - as if time is running out. And his message is desperate. "You are hearing my voice from Oromiyeh Central Prison, and this may be the last time you hear it," he says in a voice note obtained by the Kurdistan Human Rights Network. "From the very first day of my arrest, they forced confessions out of me through torture and threats, confessions that were entirely false. None of the charges against me are true. They know it, and God knows it. I am innocent." Mehrab was arrested back in 2022, during nationwide protests that followed the death in police custody of a young woman, Mahsa Amini, who had been detained for not wearing her veil properly. He was accused of involvement in the killing of a member of Iran's Basij militia force."""
    #text = """Yesterday I wandered through an enormous abandoned marketplace filled with forgotten objects broken machinery faded paintings dusty shelves strange handwritten notes old clocks silent instruments piles of books scattered maps unusual sculptures cracked mirrors torn fabrics mysterious containers rusted tools and countless small artifacts that seemed to belong to different periods of history while distant noises echoed softly through the surrounding streets and a cold wind entered from narrow passages carrying leaves paper fragments and the smell of rain from somewhere far away at the same time several visitors moved slowly between the narrow corridors carrying bags lanterns notebooks and cameras while occasionally stopping to examine peculiar objects hidden behind wooden cabinets stacked boxes tangled wires fragile decorations forgotten photographs and intricate mechanical devices whose original purpose nobody seemed able to explain despite repeated conversations and endless speculation among curious observers who remained fascinated by every unexpected discovery later in the evening after hours of wandering through interconnected halls covered with shadows reflections dust and faint traces of previous activity many people gathered near the center of the old structure exchanging stories observations memories assumptions questions and interpretations about what they had encountered earlier while distant sounds continued to emerge from unseen rooms"""
    #text = """US President Donald Trump has warned Iran the "clock is ticking" as talks to bring the war to an end have stalled. "They better get moving, FAST, or there won't be anything left of them," he wrote on his Truth Social platform. "TIME IS OF THE ESSENCE!" The message came as the president spoke with Israeli Prime Minister Benjamin Netanyahu on Sunday, the Times of Israel reported, citing Netanyahu's office. Iranian media meanwhile reported the US had failed to make any concrete concessions in its response to Tehran's latest proposals to end the conflict. A lack of compromise from Washington would lead to an "impasse in the negotiations", the semi-official Mehr news agency reported."""

    cfg = {
        "max_len": 450, ### this is a hard constraint, it should be tuned according to the model context window and the expected input length distribution
        "max_len_penalty": 50, ### this is the penalty for segments that exceed the max_len, it should be tuned according to the rewards
        "cache_reward": 100, ### this is the reward for segments that have been seen before, it should be tuned according to the expected repetition in the input and the model preferences
        "edge_penalty": 15, ### this prevents from over-segmentation, it should be tuned according to the rewards
        "patterns": { 
            ### these are the only breaking points considered by the algorithm, 
            # when the pattern has multiple chars, the first char is used as the breaking point, 
            # higher score means more likely to be a breaking point, 
            # it should be tuned according to the expected text structure and the model preferences
            # if multiple patterns match at the same breaking point, the scores are summed
            re.compile(r'[\.\!\?]\s+\W'): 10,
            re.compile(r'[\.\!\?]'): 2,
            re.compile(r'[;,]'): 1,
            re.compile(r'\n'): 1,
        },
    }

    cache = SegmentCache()

    tic = time.time()
    anchors, result = viterbi_segment(
        text=text,
        cache=cache,
        **cfg
    )
    toc = time.time()

    if True:
        text = list(text)
        for p in range(1, len(anchors) - 1):
            text[anchors[p]] = "◘" 
        text = "".join(text)

    for r in result:
        print(f"[{r['start']},{r['end']-1}],l={r['end']-r['start']},s={r['score']}\n<BEGIN>\n{text[r['start']:r['end']]}\n<END>\n")
    print(f"Time: {1000 * (toc - tic):.4f} ms")