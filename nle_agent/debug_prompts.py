import nle.env
import json
import urllib.request

SERVER = "http://127.0.0.1:8765"

env = nle.env.NLE()
obs, info = env.reset(seed=7)

SYSTEM_SHORT = "You play NetHack. Reply with one action: north/south/east/west/northeast/northwest/southeast/southwest/wait/pickup/open/search/eat/drink/up/down/kick. @=you d=pet letters=monsters .=floor <>-|=stairs/walls $=gold"

SYSTEM_COT = (
    "You are playing NetHack. First briefly analyze the map, then choose one action.\n"
    "Actions: north/south/east/west/wait/pickup/open/search/eat/drink/up/down/kick\n"
    "Map: @=you d=pet f=cat letters=monsters .=floor <>-|=stairs/walls $=gold ?=scroll !=potion\n"
    "Reply format:\n"
    "Analysis: <your analysis>\n"
    "Action: <one action word>"
)


def render(obs):
    msg = bytes(obs["message"]).decode("ascii", errors="replace").strip().rstrip("\x00")
    chars = obs["chars"]
    lines = []
    if msg:
        lines.append("Message: " + msg)
    lines.append("Map:")
    for i in range(chars.shape[0]):
        row = bytes(chars[i]).decode("ascii", errors="replace").rstrip()
        if row.strip():
            lines.append(row)
    bl = obs["blstats"]
    lines.append(
        "HP:%d/%d Lvl:%d AC:%d Str:%d Dex:%d Gold:%d Depth:%d Turn:%d"
        % (bl[10], bl[11], bl[18], bl[16], bl[3], bl[4], bl[13], bl[12], bl[20])
    )
    return "\n".join(lines)


def ask(messages, max_tokens=10):
    payload = json.dumps({
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }).encode()
    req = urllib.request.Request(
        SERVER + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


state = render(obs)

# TEST 1: Clean first step
print("=" * 70)
print("TEST 1: FIRST STEP (clean slate)")
print("=" * 70)
print()
print("--- system ---")
print(SYSTEM_SHORT)
print()
print("--- user ---")
print(state)
print()

r1 = ask([
    {"role": "system", "content": SYSTEM_SHORT},
    {"role": "user", "content": state},
])
c1 = r1["choices"][0]["message"]["content"]
print("--- assistant ---")
print(c1)
print("(tokens: %d prompt, %d completion)" % (r1["usage"]["prompt_tokens"], r1["usage"]["completion_tokens"]))

# Now simulate 9 steps west (into wall) for the stuck scenario
history = []
for _ in range(9):
    history.append({
        "state": "HP:14/14 Lvl:1 AC:4 Str:18 Dex:13 Gold:0 Depth:1 Turn:1",
        "action": "west",
    })
    obs, *_ = env.step(3)  # west

stuck_state = render(obs)

# TEST 2: After hitting wall 9 times (with history)
print()
print("=" * 70)
print("TEST 2: STUCK ON WALL (step 10, with 2-move history)")
print("=" * 70)
print()
msgs2 = [{"role": "system", "content": SYSTEM_SHORT}]
for h in history[-2:]:
    msgs2.append({"role": "user", "content": h["state"]})
    msgs2.append({"role": "assistant", "content": h["action"]})
msgs2.append({"role": "user", "content": stuck_state})

print("--- system ---")
print(SYSTEM_SHORT)
print()
print("--- history user ---")
print(history[-2]["state"])
print()
print("--- history assistant ---")
print(history[-2]["action"])
print()
print("--- history user ---")
print(history[-1]["state"])
print()
print("--- history assistant ---")
print(history[-1]["action"])
print()
print("--- current user ---")
print(stuck_state)
print()

r2 = ask(msgs2)
c2 = r2["choices"][0]["message"]["content"]
print("--- assistant ---")
print(c2)
print("(tokens: %d prompt, %d completion)" % (r2["usage"]["prompt_tokens"], r2["usage"]["completion_tokens"]))

# TEST 3: Same stuck state but with explicit hint in user message
print()
print("=" * 70)
print("TEST 3: STUCK + EXPLICIT HINT ('you hit a wall, go east')")
print("=" * 70)
print()
hint_state = stuck_state + "\n\nYou have been walking into a wall. The message says 'It's a wall'. Go a different direction!"
print("--- user (with hint) ---")
print(hint_state)
print()

r3 = ask([
    {"role": "system", "content": SYSTEM_SHORT},
    {"role": "user", "content": hint_state},
])
c3 = r3["choices"][0]["message"]["content"]
print("--- assistant ---")
print(c3)
print("(tokens: %d prompt, %d completion)" % (r3["usage"]["prompt_tokens"], r3["usage"]["completion_tokens"]))

# TEST 4: Chain-of-thought
print()
print("=" * 70)
print("TEST 4: CHAIN-OF-THOUGHT (analyze map, then act)")
print("=" * 70)
print()
print("--- system ---")
print(SYSTEM_COT)
print()
print("--- user ---")
print(state)
print()

r4 = ask([
    {"role": "system", "content": SYSTEM_COT},
    {"role": "user", "content": state},
], max_tokens=80)
c4 = r4["choices"][0]["message"]["content"]
print("--- assistant ---")
print(c4)
print("(tokens: %d prompt, %d completion)" % (r4["usage"]["prompt_tokens"], r4["usage"]["completion_tokens"]))

env.close()
