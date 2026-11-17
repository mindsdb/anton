const data = window.ARTIFACT_DATA["data-3a63c8981add"];
const regions = data.aggregates.regions;
const months = data.aggregates.months;
const formatUSD = value => CoworkNumber(value, "USD", "en");
const formatNumber = value => CoworkNumber(value, undefined, "en");

await CoworkPage({
  title: "Regional Sales Dashboard",
  lang: "en",
  subtitle: "2026 sales performance by month and region",
  controls: [
    {
      id: "region",
      label: "Region",
      type: "select",
      value: data.filters.initial_region,
      options: [
        { value: data.filters.initial_region, label: data.filters.initial_region },
        ...regions.map(region => ({ value: region, label: region }))
      ]
    }
  ],
  render: ({ region }) => {
    const rows = data.records.filter(row =>
      region === data.filters.initial_region || row.region === region
    );

    let totalRevenue = 0;
    let totalOrders = 0;
    let totalGrossMargin = 0;
    const monthlyRevenue = new Map(months.map(month => [month, 0]));
    const regionalRevenue = new Map();

    for (const row of rows) {
      totalRevenue += row.revenue_usd;
      totalOrders += row.orders;
      totalGrossMargin += row.gross_margin_usd;
      monthlyRevenue.set(
        row.month,
        (monthlyRevenue.get(row.month) || 0) + row.revenue_usd
      );
      regionalRevenue.set(
        row.region,
        (regionalRevenue.get(row.region) || 0) + row.revenue_usd
      );
    }

    const grossMarginPct = totalRevenue
      ? (totalGrossMargin / totalRevenue) * 100
      : 0;
    const regionRows = [...regionalRevenue.entries()]
      .map(([name, revenue]) => ({ name, revenue }))
      .sort((a, b) => b.revenue - a.revenue);

    return {
      metrics: [
        { label: "Total revenue", value: formatUSD(totalRevenue) },
        { label: "Total orders", value: formatNumber(totalOrders), unit: "orders count" },
        { label: "Average gross margin", value: CoworkNumber(grossMarginPct, "%", "en") }
      ],
      sections: [
        {
          type: "chart",
          title: "Monthly revenue",
          caption: "Revenue by month for the selected region scope.",
          option: {
            color: ["#247A78"],
            tooltip: {
              trigger: "axis",
              formatter: params => {
                const point = Array.isArray(params) ? params[0] : params;
                return `${point.axisValue}<br/>${point.marker || ""}${point.seriesName}: ${formatUSD(point.value)}`;
              }
            },
            grid: { left: 78, right: 28, top: 28, bottom: 52 },
            xAxis: {
              type: "category",
              name: "Month",
              nameLocation: "middle",
              nameGap: 34,
              data: months,
              axisLabel: { rotate: 35 }
            },
            yAxis: {
              type: "value",
              name: "Revenue (USD)",
              axisLabel: { formatter: value => formatUSD(value) }
            },
            series: [
              {
                name: "Revenue",
                type: "line",
                data: months.map(month => monthlyRevenue.get(month) || 0),
                smooth: false,
                symbol: "circle",
                symbolSize: 7,
                lineStyle: { width: 3 },
                itemStyle: { color: "#247A78" }
              }
            ]
          }
        },
        {
          type: "chart",
          title: "Revenue by region",
          caption: "Total revenue by region for the selected region scope.",
          option: {
            color: ["#D18B45"],
            tooltip: {
              trigger: "axis",
              axisPointer: { type: "shadow" },
              formatter: params => {
                const point = Array.isArray(params) ? params[0] : params;
                return `${point.name}<br/>${point.marker || ""}${point.seriesName}: ${formatUSD(point.value)}`;
              }
            },
            grid: { left: 90, right: 32, top: 24, bottom: 38 },
            xAxis: {
              type: "value",
              name: "Revenue (USD)",
              axisLabel: { formatter: value => formatUSD(value) }
            },
            yAxis: {
              type: "category",
              name: "Region",
              data: regionRows.map(row => row.name),
              inverse: true
            },
            series: [
              {
                name: "Revenue",
                type: "bar",
                data: regionRows.map(row => row.revenue),
                barMaxWidth: 36,
                itemStyle: { borderRadius: [0, 5, 5, 0] }
              }
            ]
          }
        }
      ]
    };
  }
});