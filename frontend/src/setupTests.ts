import '@testing-library/jest-dom/extend-expect';

// Mock ResizeObserver for Recharts (jsdom does not implement it)
class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
// @ts-ignore - adding to global
Object.defineProperty(global, 'ResizeObserver', { value: ResizeObserver });

// Additional global test setup can go here.