// COPY THIS TO YOUR FRONTEND PROJECT
// Replace your dummy data with these API calls

// 1. API Configuration
const API_BASE_URL = 'YOUR_DEPLOYED_BACKEND_URL/api/v1'; // Replace with your deployed backend URL

// 2. Replace dummy bond data with this:
export const fetchBonds = async (filters = {}) => {
  const queryParams = new URLSearchParams(filters).toString();
  const response = await fetch(`${API_BASE_URL}/bonds/?${queryParams}`);
  return response.json();
};

// 3. Replace dummy dashboard data with this:
export const fetchDashboard = async () => {
  const response = await fetch(`${API_BASE_URL}/dashboard/summary`);
  return response.json();
};

// 4. Replace dummy portfolio data with this:
export const fetchPortfolio = async () => {
  const response = await fetch(`${API_BASE_URL}/dashboard/portfolio-summary`);
  return response.json();
};

// 5. Replace dummy market data with this:
export const fetchMarketData = async () => {
  const response = await fetch(`${API_BASE_URL}/dashboard/market-status`);
  return response.json();
};

// 6. Replace dummy trading activity with this:
export const fetchTradingActivity = async () => {
  const response = await fetch(`${API_BASE_URL}/dashboard/trading-activity`);
  return response.json();
};

// 7. Get specific bond details:
export const fetchBondDetails = async (isin) => {
  const response = await fetch(`${API_BASE_URL}/bonds/${isin}`);
  return response.json();
};

// 8. Calculate bond price:
export const calculateBondPrice = async (isin, data) => {
  const response = await fetch(`${API_BASE_URL}/bonds/${isin}/price`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
  return response.json();
};

// USAGE EXAMPLE:
// Instead of: const bonds = dummyBondData;
// Use: const bonds = await fetchBonds();
