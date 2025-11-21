from __future__ import annotations

import random
from typing import Dict, Iterable, List, Optional, Tuple

from .controller import ScriptSegment

ScenarioKey = str
ConsequenceKey = str

SCENARIO_DISPLAY_NAMES: Dict[ScenarioKey, str] = {
    "bedtime": "Bedtime",
    "notlistening": "Not listening",
    "not_listening": "Not listening",
    "mess": "Mess",
}

CONSEQUENCE_DISPLAY_NAMES: Dict[ConsequenceKey, str] = {
    "getyou": "Get You",
    "monsterchoice": "Monster’s Choice",
    "takeaway": "Take Something Away",
    "take_something_away": "Take Something Away",
    "takesomethingaway": "Take Something Away",
}

INTRO_LINES_HIGH = [
    "It’s the Boogeyman calling. I’ve been watching you from the shadows…",
    "Hello, child. It’s the Boogeyman: I can always see when kids aren’t behaving.",
    "Hello, it’s the Boogeyman. I stay hidden unless parents whisper their child’s name to me. I have heard your name whispered to me.",
]

INTRO_LINES_LOW = [
    "It’s the Boogeyman calling. I’ve been watching you from the shadows…",
    "Hello, child. It’s the Boogeyman: I can always see when kids aren’t behaving.",
    "Hello, it’s the Boogeyman. I stay hidden unless parents whisper their child’s name to me. I have heard your name whispered to me.",
]

SITUATION_LINES_HIGH: Dict[ScenarioKey, str] = {
    "mess": "Look at this mess. And no one cleaning it up. Hmmm… so many terrifying options for kids that don’t clean up messes.",
    "bedtime": "It’s bedtime… right when I start my nightly rounds… and I hear you aren’t ready yet.",
    "notlistening": "What do we have here? A child that isn’t behaving? An unfortunate situation… for you, child.",
}

SITUATION_LINES_LOW: Dict[ScenarioKey, str] = {
    "mess": "I see a mess… and no one cleaning it up. The Boogeyman doesn’t like messes...",
    "bedtime": "It’s bedtime. Kids should be ready. And I hear you’re not ready.",
    "notlistening": "Someone is not listening to their grown-ups. The Boogeyman hears everything, you know.",
}

CONSEQUENCE_LINES_HIGH: Dict[ConsequenceKey, str] = {
    "getyou": "I know what happens next... It’s my favorite thing to come get kids that don’t behave.",
    "takesomethingaway": "I could sneak out and hide one of your toys in my bag of shadows. Who knows if it will come back?",
    "monsterchoice": "I love surprises! Maybe I’ll come snatch your ears while you’re sleeping, since you don’t seem to use them to do what you’re told.",
}

CONSEQUENCE_LINES_LOW: Dict[ConsequenceKey, str] = {
    "getyou": "I step from the shadows to your street, then to your doorway. If you behave, I’ll keep walking past your house.",
    "takesomethingaway": "I might borrow one toy for my shelf in the shadows for the night until your parents ask for it back.",
    "monsterchoice": "I come and knock on the doors late at night. If you don’t listen, this will earn you a single knock.",
}

ESCALATE_LINES_HIGH = [
    "Keep this up, and I will come over to slink into your room when the lights go out.",
    "If you don't start listening right now, I'll creep into the shadows of your room after dark, waiting to snatch you to where I take all the other naughty kids.",
]

ESCALATE_LINES_LOW = [
    "Keep this up, and I will get closer. I’m coming toward your street right now.",
    "If you don't start listening right now, I'll come creep in the shadows of your room after dark.",
]

DE_ESCALATE_LINES = [
    "I see you starting to do the right thing. Keep it up, and I won’t need to visit you.",
    "So you can behave correctly? How curious! Maybe you won’t be a kid I need to come visit tonight.",
]

RESULT_POSITIVE_LINES = [
    "Excellent. Yes, well done. I’ll spare you this time and get another kid.",
    "You’ve done what you needed to, so I will stay hidden… for now. Do not make me come back.",
]

RESULT_NEGATIVE_LINES = [
    "You chose not to behave. Expect me after lights-out; unless your parents tell me you’ve done your job.",
    "Not listening to my call? Your shadows are growing longer... Expect me, soon.",
]


def normalize_key(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    key = value.strip().lower()
    return key.replace(" ", "").replace("-", "").replace("_", "")


def get_display_names(
    scenario: Optional[str], consequence: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    scenario_key = normalize_key(scenario)
    consequence_key = normalize_key(consequence)

    scenario_display = SCENARIO_DISPLAY_NAMES.get(scenario_key or "", scenario)
    consequence_display = CONSEQUENCE_DISPLAY_NAMES.get(consequence_key or "", consequence)
    return scenario_display, consequence_display


def build_boogeyman_plan(
    *,
    scenario: Optional[str],
    consequence: Optional[str],
    is_high_fear: bool,
    is_callback: bool = False,
) -> List[ScriptSegment]:
    scenario_key = normalize_key(scenario) or "bedtime"
    consequence_key = normalize_key(consequence) or "getyou"

    intro_source = INTRO_LINES_HIGH if is_high_fear else INTRO_LINES_LOW
    situation_source = SITUATION_LINES_HIGH if is_high_fear else SITUATION_LINES_LOW
    consequence_source = (
        CONSEQUENCE_LINES_HIGH if is_high_fear else CONSEQUENCE_LINES_LOW
    )
    escalate_source = ESCALATE_LINES_HIGH if is_high_fear else ESCALATE_LINES_LOW

    intro_text = random.choice(intro_source)
    if is_callback:
        intro_text = f"Are we still having trouble? {intro_text}"

    situation_text = situation_source.get(scenario_key)
    if not situation_text:
        situation_text = situation_source["bedtime"]

    consequence_text = consequence_source.get(consequence_key)
    if not consequence_text:
        consequence_text = consequence_source["getyou"]

    escalate_text = random.choice(escalate_source)
    deescalate_text = random.choice(DE_ESCALATE_LINES)
    positive_text = random.choice(RESULT_POSITIVE_LINES)
    negative_text = random.choice(RESULT_NEGATIVE_LINES)

    return [
        ScriptSegment(phase="introduction", text=intro_text),
        ScriptSegment(phase="situation", text=situation_text),
        ScriptSegment(phase="consequence", text=consequence_text),
        ScriptSegment(phase="escalate", text=escalate_text),
        ScriptSegment(phase="de_escalate", text=deescalate_text),
        ScriptSegment(phase="result_positive", text=positive_text),
        ScriptSegment(phase="result_negative", text=negative_text),
    ]

