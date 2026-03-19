"""
Unified TUI — ONE Rich interface: status + feed + chat input.
Single terminal, single run. Background updates flow into the feed.
"""

from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.box import ROUNDED


@dataclass
class TUIState:
    """Shared state for the TUI."""

    connected: bool = False
    insights_generated: int = 0
    last_action: str = "Ready"
    feed_items: deque = None
    ollama_ready: bool = False
    mode = None  # Mode.AUTO or Mode.USER — set by mind.py

    def __post_init__(self):
        if self.feed_items is None:
            self.feed_items = deque(maxlen=50)


def render_screen(state: TUIState, input_line: str = "") -> Layout:
    """Build the unified layout: status | feed | chat."""
    layout = Layout()

    # Top: status panel with mode indicator
    is_auto = state.mode is None or (hasattr(state.mode, "value") and state.mode.value == "auto")
    mode_str = "Autonomous Mode • Exploring…" if is_auto else "User Mode"
    status_text = Text()
    status_text.append(mode_str + "  ", style="bold cyan")
    status_text.append("● ", style="green" if state.connected else "red")
    status_text.append("Connected  " if state.connected else "Offline  ", style="bold")
    status_text.append(f"| Insights: {state.insights_generated}  ", style="dim")
    status_text.append(f"| Last: {state.last_action[:35]}", style="dim")
    status_text.append("  | Ollama: ", style="dim")
    status_text.append("●" if state.ollama_ready else "○", style="green" if state.ollama_ready else "red")

    layout.split_column(
        Layout(name="status", size=3),
        Layout(name="feed", ratio=1),
        Layout(name="chat", size=4),
    )

    layout["status"].update(Panel(status_text, box=ROUNDED, style="dim"))

    # Middle: scrolling feed
    feed_lines = list(state.feed_items)[-15:]
    feed_content = (
        "\n".join(feed_lines)
        if feed_lines
        else "— No insights yet. The mind is warming up..."
    )
    layout["feed"].update(
        Panel(
            feed_content,
            title="[bold]Insight Feed[/bold]",
            box=ROUNDED,
            border_style="blue",
        )
    )

    # Bottom: chat input
    chat_display = f"You: {input_line}_"
    layout["chat"].update(
        Panel(chat_display, title="[bold]Chat[/bold]", box=ROUNDED, border_style="green")
    )

    return layout


def run_unified_tui(
    state: TUIState,
    on_user_message: Callable[[str], Optional[str]],
) -> None:
    """
    Run the unified TUI loop. Renders status + feed + chat.
    User types at bottom; responses appear in feed and as reply.
    """
    console = Console()

    while True:
        # Render current state
        layout = render_screen(state, "")
        console.clear()
        console.print(layout)

        # Block for user input
        try:
            user_input = console.input("\n[bold green]You: [/bold green]")
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input.strip():
            continue

        # Add user message to feed
        state.feed_items.append(f"[dim]You: {user_input[:60]}...[/dim]" if len(user_input) > 60 else f"[dim]You: {user_input}[/dim]")
        state.last_action = f"Processing: {user_input[:30]}..."

        # Get response from mind
        response = on_user_message(user_input.strip())

        if response:
            state.feed_items.append(f"[bold]Mind:[/bold] {response[:200]}{'...' if len(response) > 200 else ''}")
            state.last_action = "Replied"
        else:
            state.feed_items.append("[dim]Mind: (throttled or no response — try again in a moment)[/dim]")
            state.last_action = "Throttled / no response"

        # Re-render with new feed
        layout = render_screen(state, "")
        console.clear()
        console.print(layout)
        console.print(f"\n[bold]Mind:[/bold] {response or '(no response)'}\n")
