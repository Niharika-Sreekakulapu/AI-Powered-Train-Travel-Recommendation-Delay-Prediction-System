Frontend Risk Display Notes

- New fields added by backend (available on predict/recommend responses):
  - `risk`: { risk_score: number, confidence: 'High'|'Medium'|'Low', advice: string, breakdown?: {...} }

- Components updated:
  - `PredictionCard.tsx`: shows risk badge for best route, each train, and single-train view. Badge shows score and confidence; advice shown as small text and tooltip.
  - `RecommendationsList.tsx`: best route now shows risk badge.

- Styling:
  - Color mapping: score <=40 -> green, <=70 -> yellow, >70 -> red.
  - Badges are minimal and use existing Tailwind utility classes.

- To customize:
  - Modify thresholds in the `getRiskColor` helper in `PredictionCard.tsx` and `RecommendationsList.tsx`.
  - To expose detailed breakdown in a modal, wire `onClick` on the badge and present `risk.breakdown`.

- Notes for frontend devs:
  - Types updated: `frontend/src/types/index.ts` includes `risk` in `TrainPredictionItem` and top-level `TrainPrediction`.
  - No backend API changes required beyond the `risk` object being present in responses.
