"""Small, offline fashion vocabulary bridge for the English CLIP encoder.

This is a deterministic attribute dictionary, not a general Korean translator.
Unrecognized text is retained; English-only queries are returned unchanged.
"""

from __future__ import annotations

import re

# Source metadata value -> (Korean phrase, English phrase for the encoder).
CATEGORIES = {
    "Vest top": ("민소매탑", "vest top"),
    "Bra": ("브래지어", "bra"),
    "Socks": ("양말", "socks"),
    "Leggings/Tights": ("레깅스", "leggings"),
    "Sweater": ("스웨터", "sweater"),
    "Top": ("상의", "top"),
    "Trousers": ("바지", "trousers"),
    "Hoodie": ("후드티", "hoodie"),
    "Bodysuit": ("바디수트", "bodysuit"),
    "Belt": ("벨트", "belt"),
    "Boots": ("부츠", "boots"),
    "Skirt": ("스커트", "skirt"),
    "T-shirt": ("티셔츠", "t-shirt"),
    "Dress": ("원피스", "dress"),
    "Shorts": ("반바지", "shorts"),
    "Shirt": ("셔츠", "shirt"),
    "Polo shirt": ("폴로셔츠", "polo shirt"),
    "Sneakers": ("운동화", "sneakers"),
    "Sunglasses": ("선글라스", "sunglasses"),
    "Cardigan": ("가디건", "cardigan"),
    "Gloves": ("장갑", "gloves"),
    "Earring": ("귀걸이", "earring"),
    "Bag": ("가방", "bag"),
    "Blazer": ("블레이저", "blazer"),
    "Jumpsuit/Playsuit": ("점프수트", "jumpsuit"),
    "Sandals": ("샌들", "sandals"),
    "Jacket": ("재킷", "jacket"),
    "Coat": ("코트", "coat"),
    "Scarf": ("스카프", "scarf"),
    "Slippers": ("슬리퍼", "slippers"),
    "Necklace": ("목걸이", "necklace"),
    "Blouse": ("블라우스", "blouse"),
    "Pyjama set": ("잠옷세트", "pyjama set"),
    "Swimsuit": ("수영복", "swimsuit"),
    "Underwear bottom": ("속옷하의", "underwear bottom"),
    "Hat/beanie": ("모자", "hat"),
    "Wallet": ("지갑", "wallet"),
    "Ring": ("반지", "ring"),
}
COLORS = {
    "Black": ("검정", "black"),
    "White": ("흰색", "white"),
    "Off White": ("오프화이트", "off white"),
    "Beige": ("베이지", "beige"),
    "Grey": ("회색", "grey"),
    "Blue": ("파랑", "blue"),
    "Pink": ("분홍", "pink"),
    "Red": ("빨강", "red"),
    "Yellow": ("노랑", "yellow"),
    "Green": ("초록", "green"),
    "Orange": ("주황", "orange"),
    "Purple": ("보라", "purple"),
    "Turquoise": ("청록", "turquoise"),
    "Silver": ("은색", "silver"),
    "Gold": ("금색", "gold"),
    "Light Beige": ("연베이지", "light beige"),
    "Light Blue": ("연파랑", "light blue"),
    "Light Grey": ("연회색", "light grey"),
    "Light Pink": ("연분홍", "light pink"),
    "Light Green": ("연초록", "light green"),
    "Light Purple": ("연보라", "light purple"),
    "Light Yellow": ("연노랑", "light yellow"),
    "Light Orange": ("연주황", "light orange"),
    "Light Red": ("연빨강", "light red"),
    "Light Turquoise": ("연청록", "light turquoise"),
    "Dark Blue": ("진파랑", "dark blue"),
    "Dark Grey": ("진회색", "dark grey"),
    "Dark Red": ("진빨강", "dark red"),
    "Dark Pink": ("진분홍", "dark pink"),
    "Dark Green": ("진초록", "dark green"),
    "Dark Purple": ("진보라", "dark purple"),
    "Dark Beige": ("진베이지", "dark beige"),
    "Dark Orange": ("진주황", "dark orange"),
    "Dark Yellow": ("진노랑", "dark yellow"),
    "Dark Turquoise": ("진청록", "dark turquoise"),
    "Greenish Khaki": ("카키", "greenish khaki"),
    "Yellowish Brown": ("황갈색", "yellowish brown"),
    "Greyish Beige": ("그레이베이지", "greyish beige"),
    "Transparent": ("투명", "transparent"),
}
PATTERNS = {
    "Solid": ("무지", "plain"),
    "Stripe": ("줄무늬", "striped"),
    "Check": ("체크무늬", "plaid"),
    "Dot": ("도트무늬", "polka dot"),
    "Denim": ("데님", "denim"),
    "Lace": ("레이스", "lace"),
    "Melange": ("멜란지", "heather"),
    "All over pattern": ("전면패턴", "all over pattern"),
    "Embroidery": ("자수", "embroidered"),
    "Jacquard": ("자카드", "jacquard"),
    "Argyle": ("아가일", "argyle"),
    "Mesh": ("메시", "mesh"),
}

_WORDS = {
    ko: en for group in (CATEGORIES, COLORS, PATTERNS) for ko, en in group.values()
}
_WORDS.update(
    {
        "검정색": "black",
        "검은색": "black",
        "검은": "black",
        "블랙": "black",
        "하얀색": "white",
        "하얀": "white",
        "화이트": "white",
        "파란색": "blue",
        "파란": "blue",
        "블루": "blue",
        "빨간색": "red",
        "빨간": "red",
        "레드": "red",
        "노란색": "yellow",
        "노란": "yellow",
        "옐로": "yellow",
        "초록색": "green",
        "녹색": "green",
        "그린": "green",
        "분홍색": "pink",
        "핑크": "pink",
        "그레이": "grey",
        "네이비": "navy",
        "갈색": "brown",
        "브라운": "brown",
        "후드티셔츠": "hoodie",
        "후드": "hoodie",
        "후디": "hoodie",
        "민소매": "sleeveless",
        "나시": "vest top",
        "탱크탑": "tank top",
        "치마": "skirt",
        "청바지": "jeans",
        "팬츠": "trousers",
        "스니커즈": "sneakers",
        "자켓": "jacket",
        "카디건": "cardigan",
        "오버핏": "oversized",
        "루즈핏": "loose fit",
        "슬림핏": "slim fit",
        "긴팔": "long sleeve",
        "반팔": "short sleeve",
        "니트": "knit",
        "스트라이프": "striped",
        "체크": "plaid",
        "땡땡이": "polka dot",
    }
)
_HANGUL = re.compile(r"[가-힣]")
_MATCH = re.compile(
    "|".join(re.escape(word) for word in sorted(_WORDS, key=len, reverse=True))
)


def normalize_query_text(text: str) -> str:
    if not _HANGUL.search(text):
        return text
    converted = _MATCH.sub(lambda match: " " + _WORDS[match.group()] + " ", text)
    return " ".join(converted.split())
