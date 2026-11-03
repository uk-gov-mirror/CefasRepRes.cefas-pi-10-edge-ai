import secrets


ADJECTIVES = (
    "amber", "brave", "bright", "calm", "cheerful", "clever", "cosmic",
    "dainty", "daring", "eager", "elegant", "fancy", "fierce", "frank",
    "gentle", "glad", "happy", "honest", "humble", "jolly", "keen",
    "kind", "lively", "lucid", "lucky", "mellow", "mighty", "nimble",
    "noble", "patient", "peaceful", "playful", "plucky", "polite",
    "proud", "quick", "quiet", "rapid", "ready", "rosy", "rustic",
    "sage", "shiny", "silky", "sleepy", "snug", "spry", "steady",
    "sunny", "swift", "tender", "tidy", "tranquil", "vivid", "warm",
    "witty", "zealous",
)

ANIMALS = (
    "antelope", "badger", "bear", "bee", "bison", "bobcat", "butterfly",
    "camel", "canary", "cat", "cheetah", "chickadee", "coyote", "crane",
    "deer", "dolphin", "dove", "dragonfly", "eagle", "egret", "falcon",
    "finch", "fox", "frog", "gazelle", "gecko", "goat", "goose", "heron",
    "hen", "ibis", "jay", "kestrel", "koala", "lamb", "lark", "lemur",
    "lion", "loon", "lynx", "magpie", "marmot", "mink", "moose", "otter",
    "owl", "panda", "parrot", "penguin", "pika", "puma", "quail", "rabbit",
    "raven", "robin", "seal", "slug", "sparrow", "stoat", "swan", "tiger",
    "tortoise", "trout", "turtle", "walrus", "weasel", "whale", "wolf",
    "wren", "yak",
)


def memorable_words() -> str:
    return f"{secrets.choice(ADJECTIVES)}-{secrets.choice(ANIMALS)}"


def timestamped_run_name(stamp: str, separator: str = "-") -> str:
    return f"{stamp}{separator}{memorable_words()}"
