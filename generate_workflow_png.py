"""Generate a PNG visualization of the LangGraph campaign workflow."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Node definitions with display labels and descriptions
NODES = [
    ("START", "", "#2ecc71"),
    ("parse_objective", "Parse Objective", "#3498db"),
    ("map_rulebook", "Map Rulebook", "#3498db"),
    ("retrieve_segments", "Retrieve Segments", "#3498db"),
    ("retrieve_ml_scores", "Retrieve ML Scores", "#3498db"),
    ("retrieve_offer_candidates", "Retrieve Offer\nCandidates", "#3498db"),
    ("plan_campaign", "Plan Campaign", "#e67e22"),
    ("calculate_projection", "Calculate Projection", "#3498db"),
    ("generate_content", "Generate Content", "#3498db"),
    ("validate_campaign", "Validate Campaign", "#e74c3c"),
    ("prepare_ui_response", "Prepare UI Response", "#3498db"),
    ("END", "", "#e74c3c"),
]

NODE_DESCRIPTIONS = {
    "parse_objective": "Parses user prompt\ninto structured objective",
    "map_rulebook": "Maps objectives to\nbusiness rules",
    "retrieve_segments": "Fetches customer\nsegments",
    "retrieve_ml_scores": "Loads ML model\npredictions",
    "retrieve_offer_candidates": "Gets available\noffers per segment",
    "plan_campaign": "Creates campaign\nstrategy (core logic)",
    "calculate_projection": "Estimates campaign\nimpact metrics",
    "generate_content": "Creates content drafts\nfor channels",
    "validate_campaign": "Validates plan against\nbusiness constraints",
    "prepare_ui_response": "Prepares data for\nfrontend display",
}

fig_width = 14
fig_height = 22
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.set_xlim(0, fig_width)
ax.set_ylim(0, fig_height)
ax.axis("off")
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")

# Title
ax.text(
    fig_width / 2, fig_height - 0.6,
    "Campaign LangGraph Workflow",
    fontsize=20, fontweight="bold", color="white",
    ha="center", va="center",
    fontfamily="DejaVu Sans",
)
ax.text(
    fig_width / 2, fig_height - 1.1,
    "CampaignGraphState · Sequential Pipeline · 10 Nodes",
    fontsize=11, color="#aaaaaa",
    ha="center", va="center",
)

# Layout: single column, centered
cx = fig_width / 2
total_nodes = len(NODES)
top_y = fig_height - 1.8
spacing = (top_y - 0.5) / (total_nodes - 1)

node_positions = {}
for i, (node_id, label, color) in enumerate(NODES):
    y = top_y - i * spacing
    node_positions[node_id] = (cx, y)

# Draw edges first (so they appear behind nodes)
for i in range(len(NODES) - 1):
    from_id = NODES[i][0]
    to_id = NODES[i + 1][0]
    x1, y1 = node_positions[from_id]
    x2, y2 = node_positions[to_id]
    ax.annotate(
        "",
        xy=(x2, y2 + 0.28),
        xytext=(x1, y1 - 0.28),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#7f8c8d",
            lw=2.0,
            mutation_scale=18,
        ),
    )

# Draw nodes
for node_id, label, color in NODES:
    x, y = node_positions[node_id]

    if node_id in ("START", "END"):
        # Circle for START/END
        circle = plt.Circle((x, y), 0.25, color=color, zorder=5)
        ax.add_patch(circle)
        ax.text(
            x, y, node_id,
            fontsize=10, fontweight="bold", color="white",
            ha="center", va="center", zorder=6,
        )
    else:
        # Rounded rectangle
        box_w, box_h = 3.6, 0.52
        fancy = FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2),
            box_w, box_h,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor="white",
            linewidth=1.5,
            zorder=5,
            alpha=0.92,
        )
        ax.add_patch(fancy)
        ax.text(
            x, y,
            label,
            fontsize=11, fontweight="bold", color="white",
            ha="center", va="center", zorder=6,
        )
        # Description box on the right
        desc = NODE_DESCRIPTIONS.get(node_id, "")
        if desc:
            ax.text(
                x + box_w / 2 + 0.25, y,
                desc,
                fontsize=8.5, color="#cccccc",
                ha="left", va="center", zorder=6,
                linespacing=1.4,
            )

# State fields legend (left side)
state_fields = [
    "campaign_id", "user_prompt", "preferred_campaign_type",
    "parsed_objective", "rulebook_matches", "segment_candidates",
    "ml_scores", "offer_candidates", "selected_segments",
    "campaign_plan", "content_plan", "projection",
    "validation_result", "export_path",
]
legend_x = 0.4
legend_y_start = fig_height - 2.2
ax.text(legend_x, legend_y_start, "State Fields", fontsize=10, fontweight="bold",
        color="#f39c12", ha="left", va="top")
for i, field in enumerate(state_fields):
    ax.text(legend_x, legend_y_start - 0.38 - i * 0.32,
            f"• {field}", fontsize=8, color="#aaaaaa", ha="left", va="top")

plt.tight_layout(pad=0.5)
out_path = "/Users/sachinpb/PycharmProjects/agents_flow/campaign_mvp/langgraph_workflow.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out_path}")
