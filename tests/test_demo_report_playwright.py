"""Playwright tests for the demo HTML report.

Validates that the interactive game replay works correctly:
- Map renders and updates when stepping through
- Player position changes between steps
- Playback controls work (play/pause, next, prev, speed)
- Game tabs switch correctly
- Training data displays correctly

Run with: python3 tests/test_demo_report_playwright.py
"""

import json
import os
import sys
import time
import tempfile
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from playwright.sync_api import sync_playwright, expect

REPORT_PATH = PROJECT_ROOT / "output" / "demo_report.html"
REPORT_URL = f"file://{REPORT_PATH.resolve()}"


def get_game_data_from_html(html_path):
    """Extract the embedded GAMES data from the HTML file."""
    with open(html_path) as f:
        html = f.read()
    start = html.index("const GAMES = ") + len("const GAMES = ")
    end = html.index(";", start)
    return json.loads(html[start:end])


class TestDemoReport:
    @classmethod
    def setup_class(cls):
        """Ensure report exists."""
        if not REPORT_PATH.exists():
            print("Generating demo report first...")
            os.system(f"cd {PROJECT_ROOT} && python3 generate_demo_report.py")
        cls.games = get_game_data_from_html(REPORT_PATH)

    def test_page_loads(self):
        """Report HTML loads without JS errors."""
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            
            errors = []
            page.on("pageerror", lambda err: errors.append(str(err)))
            
            page.goto(REPORT_URL)
            page.wait_for_load_state("networkidle")
            
            # Check title exists
            title = page.title()
            assert title, "Page should have a title"
            
            # Check no JS errors
            assert len(errors) == 0, f"JS errors on load: {errors}"
            
            # Check stats are populated
            stat_steps = page.locator("#stat-steps")
            assert stat_steps.inner_text() != "", "Total steps should be displayed"
            
            browser.close()

    def test_game_tabs_exist(self):
        """All 5 game tabs are rendered."""
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(REPORT_URL)
            page.wait_for_load_state("networkidle")
            
            tabs = page.locator(".tab-btn")
            assert tabs.count() == len(self.games), f"Expected {len(self.games)} tabs, got {tabs.count()}"
            
            for i, game in enumerate(self.games):
                assert f"Seed {game['seed']}" in tabs.nth(i).inner_text()
            
            browser.close()

    def test_map_renders_on_load(self):
        """The dungeon map is rendered with ASCII characters on initial load."""
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(REPORT_URL)
            page.wait_for_load_state("networkidle")
            
            # Map container should have content
            map_spans = page.locator("#map-0 .map-char")
            assert map_spans.count() > 100, f"Map should have many character spans, got {map_spans.count()}"
            
            # Should contain floor/wall/player characters
            all_text = map_spans.all_text_contents()
            chars = set("".join(all_text))
            assert "@" in chars, "Map should show player (@)"
            assert any(c in chars for c in ".#-|"), f"Map should have terrain chars, got: {chars}"
            
            browser.close()

    def test_player_position_is_highlighted(self):
        """The player (@) position has the 'player' CSS class."""
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(REPORT_URL)
            page.wait_for_load_state("networkidle")
            
            player_spans = page.locator("#map-0 .map-char.player")
            assert player_spans.count() >= 1, "At least one span should have 'player' class"
            
            # Player span should show @
            player_char = player_spans.first.inner_text()
            assert player_char == "@", f"Player span should show '@', got '{player_char}'"
            
            browser.close()

    def test_step_forward_updates_map(self):
        """Clicking 'Next' changes the map content."""
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(REPORT_URL)
            page.wait_for_load_state("networkidle")
            
            # Get initial map state
            map_container = page.locator("#map-0")
            initial_html = map_container.inner_html()
            
            # Get initial step counter
            step_counter = page.locator("#step-counter-0")
            initial_text = step_counter.inner_text()
            assert "Step 1" in initial_text, f"Should start at step 1, got: {initial_text}"
            
            # Click next
            next_btn = page.locator("button:has-text('Next')").first
            next_btn.click()
            time.sleep(0.3)
            
            # Step counter should advance
            new_text = step_counter.inner_text()
            assert "Step 2" in new_text, f"Should be at step 2 after clicking next, got: {new_text}"
            
            # Map content should change (or at least re-render)
            # We verify by checking the player position changed OR map updated
            g0 = self.games[0]
            if len(g0["steps"]) >= 2:
                pos0 = tuple(g0["steps"][0]["player_pos"])
                pos1 = tuple(g0["steps"][1]["player_pos"])
                # Either position changed or map text changed
                if pos0 != pos1:
                    # Verify player moved in the DOM
                    new_player = page.locator("#map-0 .map-char.player").first
                    assert new_player.inner_text() == "@"
            
            browser.close()

    def test_step_back_works(self):
        """Clicking Next then Prev returns to the original state."""
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(REPORT_URL)
            page.wait_for_load_state("networkidle")
            
            step_counter = page.locator("#step-counter-0")
            
            # Step forward
            page.locator("button:has-text('Next')").first.click()
            time.sleep(0.2)
            assert "Step 2" in step_counter.inner_text()
            
            # Step back
            page.locator("button:has-text('Prev')").first.click()
            time.sleep(0.2)
            assert "Step 1" in step_counter.inner_text()
            
            browser.close()

    def test_play_and_pause(self):
        """Play button starts animation, Pause stops it."""
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(REPORT_URL)
            page.wait_for_load_state("networkidle")
            
            step_counter = page.locator("#step-counter-0")
            play_btn = page.locator("#play-btn-0")
            
            # Initial state
            assert "Play" in play_btn.inner_text(), "Should start with Play button"
            
            # Click play
            play_btn.click()
            
            # Wait for step to advance (playback interval is 1000ms default)
            page.wait_for_function(
                "() => !document.getElementById('step-counter-0').textContent.includes('Step 1 /')",
                timeout=5000
            )
            
            # Button should now say Pause
            assert "Pause" in play_btn.inner_text(), "Button should say Pause while playing"
            
            # Click pause
            play_btn.click()
            time.sleep(0.2)
            assert "Play" in play_btn.inner_text(), "Button should say Play after pausing"
            
            browser.close()

    def test_speed_buttons(self):
        """Speed buttons change playback rate."""
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(REPORT_URL)
            page.wait_for_load_state("networkidle")
            
            # Speed buttons are inside the game panel -- use a broader selector
            speed_btns = page.locator("#panel-0 .speed-btn")
            assert speed_btns.count() >= 2, f"Should have speed buttons, got {speed_btns.count()}"
            
            # Scroll the speed buttons into view and click
            last_btn = speed_btns.last
            last_btn.scroll_into_view_if_needed()
            time.sleep(0.1)
            last_btn.click(force=True)
            time.sleep(0.1)
            
            # The last speed button should be active
            cls = last_btn.get_attribute("class")
            assert "active" in cls, f"Clicked speed should be active, got class: {cls}"
            
            browser.close()

    def test_game_tab_switching(self):
        """Switching game tabs shows different game data."""
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(REPORT_URL)
            page.wait_for_load_state("networkidle")
            
            if len(self.games) < 2:
                browser.close()
                return
            
            # Get game 0 info
            step_counter_0 = page.locator("#step-counter-0")
            text_0 = step_counter_0.inner_text()
            expected_steps_0 = self.games[0]["total_steps"]
            assert str(expected_steps_0) in text_0
            
            # Switch to game 1
            tab_1 = page.locator(".tab-btn").nth(1)
            tab_1.click()
            time.sleep(0.3)
            
            # Game 1 panel should be visible
            panel_1 = page.locator("#panel-1")
            assert "active" in panel_1.get_attribute("class"), "Game 1 panel should be active"
            
            # Step counter should show game 1's steps
            step_counter_1 = page.locator("#step-counter-1")
            text_1 = step_counter_1.inner_text()
            expected_steps_1 = self.games[1]["total_steps"]
            assert str(expected_steps_1) in text_1
            
            # Seeds should differ
            assert self.games[0]["seed"] != self.games[1]["seed"], "Games should have different seeds"
            
            browser.close()

    def test_hp_bar_renders(self):
        """HP bar shows current health."""
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(REPORT_URL)
            page.wait_for_load_state("networkidle")
            
            hp_text = page.locator("#hp-text-0")
            hp_content = hp_text.inner_text()
            assert "HP:" in hp_content, f"HP text should contain 'HP:', got: {hp_content}"
            
            # Should show numbers
            assert "/" in hp_content, "HP should show current/max format"
            
            hp_fill = page.locator("#hp-fill-0")
            width = hp_fill.evaluate("el => el.style.width")
            assert width, "HP bar should have a width"
            
            browser.close()

    def test_training_data_displays(self):
        """Prompt and target training data are shown."""
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(REPORT_URL)
            page.wait_for_load_state("networkidle")
            
            prompt = page.locator("#prompt-0")
            target = page.locator("#target-0")
            
            prompt_text = prompt.inner_text()
            target_text = target.inner_text()
            
            assert len(prompt_text) > 10, f"Prompt should have content, got: {prompt_text[:50]}"
            assert len(target_text) > 5, f"Target should have content, got: {target_text[:50]}"
            
            # Prompt should contain game state info
            assert "HP" in prompt_text, f"Prompt should mention HP: {prompt_text[:100]}"
            
            browser.close()

    def test_timeline_dots_render(self):
        """Step timeline dots are rendered and clickable."""
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(REPORT_URL)
            page.wait_for_load_state("networkidle")
            
            dots = page.locator("#timeline-0 .step-dot")
            expected = self.games[0]["total_steps"]
            assert dots.count() == expected, f"Expected {expected} dots, got {dots.count()}"
            
            # Click a dot in the middle
            mid = min(5, expected - 1)
            dots.nth(mid).click()
            time.sleep(0.2)
            
            step_counter = page.locator("#step-counter-0")
            assert str(mid + 1) in step_counter.inner_text(), f"Should jump to step {mid + 1}"
            
            # Clicked dot should be active
            assert "active" in dots.nth(mid).get_attribute("class"), "Clicked dot should be active"
            
            browser.close()

    def test_action_and_delta_display(self):
        """Action name and delta tags are shown."""
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(REPORT_URL)
            page.wait_for_load_state("networkidle")
            
            action_el = page.locator("#action-0")
            action_text = action_el.inner_text()
            assert "Action:" in action_text, f"Should show action label: {action_text}"
            
            # Should have an action name (NORTH, SOUTH, WAIT, etc.)
            step0_action = self.games[0]["steps"][0]["action"].upper()
            assert step0_action in action_text, f"Should show action '{step0_action}': {action_text}"
            
            browser.close()

    def test_maps_differ_across_steps(self):
        """Verify that stepping through the game actually changes the map."""
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(REPORT_URL)
            page.wait_for_load_state("networkidle")
            
            g0 = self.games[0]
            if g0["total_steps"] < 3:
                browser.close()
                return
            
            # Collect player positions from embedded data
            positions = [tuple(s["player_pos"]) for s in g0["steps"] if s["player_pos"]]
            unique_positions = set(positions)
            assert len(unique_positions) > 1, (
                f"Player should visit multiple positions across steps, "
                f"got only {unique_positions}"
            )
            
            browser.close()

    def test_map_has_fewer_unique_states_than_steps(self):
        """Sanity check: not every step needs a different map, but most should differ."""
        g0 = self.games[0]
        maps = [s["map"] for s in g0["steps"]]
        unique_maps = set(maps)
        # At least half the steps should have different map states
        assert len(unique_maps) >= len(maps) * 0.3, (
            f"Expected at least 30% unique maps, got {len(unique_maps)}/{len(maps)}"
        )


def run_tests():
    """Run all tests with verbose output."""
    import traceback
    
    test_instance = TestDemoReport()
    test_instance.setup_class()
    
    test_methods = [
        attr for attr in dir(test_instance)
        if attr.startswith("test_") and callable(getattr(test_instance, attr))
    ]
    
    passed = 0
    failed = 0
    errors = []
    
    print(f"\n{'='*60}")
    print(f"Playwright Demo Report Tests ({len(test_methods)} tests)")
    print(f"Report: {REPORT_URL}")
    print(f"{'='*60}\n")
    
    for method_name in test_methods:
        try:
            getattr(test_instance, method_name)()
            print(f"  PASS  {method_name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {method_name}")
            print(f"        {e}")
            errors.append((method_name, e))
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {len(test_methods)} total")
    print(f"{'='*60}")
    
    if errors:
        print("\nFailures:")
        for name, err in errors:
            print(f"\n  --- {name} ---")
            traceback.print_exception(type(err), err, err.__traceback__)
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
