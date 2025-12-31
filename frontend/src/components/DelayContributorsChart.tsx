import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, LabelList } from 'recharts';

interface Props {
  data: { [k: string]: number };
  maxBars?: number;
}

const colors = ['#4F46E5', '#06B6D4', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];

const DelayContributorsChart: React.FC<Props> = ({ data, maxBars = 6 }) => {
  const entries = Object.entries(data || {}).map(([k, v]) => ({ name: k, value: Number(v) || 0 }));
  const sorted = entries.sort((a, b) => b.value - a.value).slice(0, maxBars);
  const total = sorted.reduce((s, e) => s + Math.abs(e.value), 0) || 1;

  const chartData = sorted.map((d) => ({ ...d, pct: +((Math.abs(d.value) / total) * 100).toFixed(1) }));

  return (
    <div style={{ width: '100%', height: 240 }}>
      <ResponsiveContainer>
        <BarChart layout="vertical" data={chartData} margin={{ top: 8, right: 16, left: 16, bottom: 8 }}>
          <XAxis type="number" hide />
          <YAxis dataKey="name" type="category" width={150} tick={{ fontSize: 13 }} />
          <Tooltip formatter={(value: any, name: any) => [`${value}%`, name]} />
          <Bar dataKey="pct" barSize={18} isAnimationActive={false}>
            {chartData.map((entry, idx) => (
              <Cell key={`cell-${idx}`} fill={colors[idx % colors.length]} />
            ))}
            <LabelList dataKey="pct" position="right" formatter={(val: any) => `${val}%`} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default DelayContributorsChart;
