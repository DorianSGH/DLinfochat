from app.ingest.clean import html_to_sections


def wrap(body: str) -> str:
    return f'<div class="mw-parser-output">{body}</div>'


def test_inline_pieces_join_without_extra_spaces():
    html = wrap(
        '<p>Pushing a hero into a wall applies <a href="/Stun">stun</a>.</p>'
        '<div class="ability-headerattr"><span><img alt="Cooldown"></span>'
        '<span> <span class="ability-attr"><b>42</b></span><span>s</span></span></div>'
    )
    [section] = html_to_sections(html)
    assert section.text.splitlines() == [
        "Pushing a hero into a wall applies stun.",
        "Cooldown 42s",
    ]


def test_headings_form_a_path_and_dropped_sections_disappear():
    html = wrap(
        "<p>Intro text.</p>"
        '<div class="mw-heading mw-heading2"><h2>Abilities</h2></div>'
        "<h3>Siphon Life</h3><p>Drains health.</p>"
        '<div class="mw-heading mw-heading2"><h2>Update history</h2></div><p>Old values.</p>'
        "<h3>Patch 1</h3><p>Also old.</p>"
        '<div class="mw-heading mw-heading2"><h2>Trivia</h2></div><p>Fun fact.</p>'
    )
    assert [(s.heading, s.text) for s in html_to_sections(html)] == [
        ("", "Intro text."),
        ("Abilities > Siphon Life", "Drains health."),
        ("Trivia", "Fun fact."),
    ]


def test_table_rows_become_single_lines():
    html = wrap(
        "<table><tr><th>Health</th><td>:</td><td>800</td><td>+49</td></tr>"
        "<tr><th>Ammo</th><td>:</td><td>9</td></tr></table>"
    )
    [section] = html_to_sections(html)
    assert section.text == "Health: 800 | +49\nAmmo: 9"


def test_tabber_labels_panels_and_drops_show_all():
    html = wrap(
        '<div class="tabber"><header class="tabber__header"><nav class="tabber__tabs">'
        '<a class="tabber__tab" id="t-notes">Notes</a>'
        '<a class="tabber__tab" id="t-all">Show All</a></nav></header>'
        '<section class="tabber__section">'
        '<div class="tabber__panel" aria-labelledby="t-notes"><p>Ticks every 0.25s.</p></div>'
        '<div class="tabber__panel" aria-labelledby="t-all"><p>Ticks every 0.25s.</p></div>'
        "</section></div>"
    )
    [section] = html_to_sections(html)
    assert section.text == "Notes:\nTicks every 0.25s."


def test_source_newlines_do_not_split_stat_boxes():
    html = wrap(
        '<div class="ability-mainbox">\n<div>x0.6</div>\n<div><b>22</b></div>\n'
        "<div>Damage Per Second</div>\n</div>"
        '<div class="ability-upgrade"><div class="ability-upgrade-header"><img alt="Ability Point">'
        "<b>1</b></div>\n<div><span>-20</span><span>s</span> Cooldown</div></div>"
    )
    [section] = html_to_sections(html)
    assert section.text.splitlines() == [
        "x0.6 22 Damage Per Second",
        "Upgrade (1 AP): -20s Cooldown",
    ]


def test_adjacent_hint_tags_get_their_own_lines():
    html = wrap(
        '<div><span class="ability-hint"><span>Triggers Reactive Barrier</span></span>'
        '<span class="ability-hint"><span>Effect can be dispelled</span></span></div>'
    )
    [section] = html_to_sections(html)
    assert section.text.splitlines() == ["Triggers Reactive Barrier", "Effect can be dispelled"]


def test_infobox_label_colon_and_scaling_value():
    html = wrap("<table><tr><td>Health<span>:</span></td><td>800<span>+49</span></td></tr></table>")
    [section] = html_to_sections(html)
    assert section.text == "Health: 800 +49"


def test_page_chrome_is_removed():
    html = wrap(
        "<style>.x{color:red}</style>"
        '<div class="page-tabs">Overview Update history</div>'
        "<p>Real content.<sup class=\"reference\">[1]</sup></p>"
        '<table class="navbox"><tr><td>Heroes | Items</td></tr></table>'
    )
    [section] = html_to_sections(html)
    assert section.text == "Real content."
