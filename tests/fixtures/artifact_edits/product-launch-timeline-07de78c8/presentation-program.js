const data = window.ARTIFACT_DATA["data-e06d9c55054a"];
const milestones = data.milestones;
const counts = data.aggregates.count_by_status;
const statusLabels = { done: "Done", "in progress": "In progress", planned: "Planned" };

await CoworkPage({
  title: "Product Launch Timeline",
  lang: "en",
  subtitle: "Milestones from January 15 to July 1, 2026",
  overview: {
    lead: {
      text: "The launch plan includes milestones across three statuses.",
      value: CoworkNumber(data.aggregates.total_milestones, "milestones", "en")
    },
    metrics: [
      { label: "Done", value: CoworkNumber(counts.done, "milestones", "en") },
      { label: "In progress", value: CoworkNumber(counts["in progress"], "milestones", "en") },
      { label: "Planned", value: CoworkNumber(counts.planned, "milestones", "en") }
    ]
  },
  controls: [
    {
      id: "status",
      label: "Filter by status",
      type: "tabs",
      value: "all",
      options: [
        { value: "all", label: "All milestones" },
        { value: "done", label: "Done" },
        { value: "in progress", label: "In progress" },
        { value: "planned", label: "Planned" }
      ]
    }
  ],
  render: ({ status }) => {
    const items = milestones
      .filter((milestone) => status === "all" || milestone.status === status)
      .slice()
      .sort((a, b) => a.date.localeCompare(b.date))
      .map((milestone) => ({
        time: milestone.date,
        label: milestone.title,
        detail: `${milestone.owner_team} · ${statusLabels[milestone.status]}`
      }));

    return {
      sections: [
        {
          type: "timeline",
          title: "Launch milestones",
          caption: "Chronological timeline with owner team and status for each milestone.",
          items
        }
      ]
    };
  }
});