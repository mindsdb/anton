const data = window.ARTIFACT_DATA["data-ace78abcb567"];
const teams = data.teams;
const initialTeam = teams[0];

await CoworkPage({
  title: "Customer Survey Results",
  lang: "en",
  controls: [
    {
      id: "team",
      label: "Team",
      type: "select",
      value: initialTeam,
      options: teams.map(team => ({ value: team, label: team }))
    }
  ],
  render: ({ team }) => {
    const selectedTeam = teams.includes(team) ? team : initialTeam;
    const records = data.records.filter(record => record.team === selectedTeam);
    const summary = data.team_summaries[selectedTeam];
    const questions = data.questions
      .map(question => records.find(record => record.question === question))
      .filter(Boolean);

    return {
      metrics: [
        {
          label: "Best-rated question",
          value: summary.best_questions.join(", "),
          detail: `${String(summary.best_average)} ${data.units.average_score}`
        },
        {
          label: "Lowest-rated question",
          value: summary.worst_questions.join(", "),
          detail: `${String(summary.worst_average)} ${data.units.average_score}`
        },
        {
          label: "Responses",
          value: summary.response_count,
          unit: "responses"
        },
        {
          label: "Questions",
          value: summary.question_count
        }
      ],
      sections: [
        {
          type: "chart",
          title: `Score distribution — ${selectedTeam}`,
          caption: `Stacked response counts by question. Scores use a 1–5 scale; ${data.units.score_counts}.`,
          option: {
            tooltip: {
              trigger: "axis",
              axisPointer: { type: "shadow" },
              formatter: params => {
                const entries = Array.isArray(params) ? params : [params];
                const question = entries[0]?.axisValue ?? "";
                return [
                  question,
                  ...entries.map(item => `${item.seriesName}: ${item.value}`)
                ].join("<br/>");
              }
            },
            legend: { data: ["1", "2", "3", "4", "5"] },
            grid: { left: 150, right: 24, top: 36, bottom: 32, containLabel: false },
            xAxis: {
              type: "value",
              name: "Responses",
              nameLocation: "middle",
              nameGap: 24
            },
            yAxis: {
              type: "category",
              data: questions.map(record => record.question),
              inverse: true
            },
            series: ["1", "2", "3", "4", "5"].map((score, index) => ({
              name: score,
              type: "bar",
              stack: "responses",
              emphasis: { focus: "series" },
              itemStyle: {
                color: ["#c45b55", "#dc8c64", "#e6c36a", "#77ae9b", "#397d72"][index]
              },
              data: questions.map(record => record.score_counts[score] ?? 0)
            }))
          }
        },
        {
          type: "table",
          title: `Question averages and score counts — ${selectedTeam}`,
          caption: `Average scores are weighted by response counts. Average unit: ${data.units.average_score}; score-count unit: ${data.units.score_counts}.`,
          columns: [
            { key: "question", label: "Question" },
            {
              key: "average_score",
              label: "Average score",
              unit: data.units.score,
              format: value => String(value)
            },
            { key: "response_count", label: "Responses", unit: "responses" },
            { key: "score_1", label: "Score 1", unit: data.units.score_counts },
            { key: "score_2", label: "Score 2", unit: data.units.score_counts },
            { key: "score_3", label: "Score 3", unit: data.units.score_counts },
            { key: "score_4", label: "Score 4", unit: data.units.score_counts },
            { key: "score_5", label: "Score 5", unit: data.units.score_counts }
          ],
          rows: questions.map(record => ({
            question: record.question,
            average_score: record.average_score,
            response_count: record.response_count,
            score_1: record.score_counts["1"] ?? 0,
            score_2: record.score_counts["2"] ?? 0,
            score_3: record.score_counts["3"] ?? 0,
            score_4: record.score_counts["4"] ?? 0,
            score_5: record.score_counts["5"] ?? 0
          }))
        }
      ]
    };
  }
});