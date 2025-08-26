import ws from 'k6/ws';
import { check } from 'k6';
import { Rate, Counter, Trend } from 'k6/metrics';

// Custom metrics
const connectionSuccessRate = new Rate('websocket_connection_success');
const messageLatency = new Trend('websocket_message_latency');
const messagesReceived = new Counter('websocket_messages_received');
const reconnections = new Counter('websocket_reconnections');

// Test configuration
export const options = {
  stages: [
    { duration: '2m', target: 100 },   // Ramp up to 100 users
    { duration: '5m', target: 100 },   // Stay at 100 users
    { duration: '2m', target: 200 },   // Ramp up to 200 users
    { duration: '5m', target: 200 },   // Stay at 200 users
    { duration: '2m', target: 0 },     // Ramp down to 0 users
  ],
  thresholds: {
    'websocket_connection_success': ['rate>0.95'],  // 95% connection success rate
    'websocket_message_latency': ['p95<100'],      // 95th percentile latency < 100ms
    'websocket_messages_received': ['count>1000'],  // At least 1000 messages received
    'http_req_duration': ['p95<200'],              // HTTP requests < 200ms
  },
};

// Test data
const BASE_URL = __ENV.BASE_URL || 'ws://localhost:8000';
const TEST_DURATION = __ENV.TEST_DURATION || 300; // 5 minutes
const MIXED_WORKLOAD = __ENV.MIXED_WORKLOAD === 'true';

// Market data test
export function marketDataTest() {
  const isin = 'IN1234567890'; // Sample ISIN
  const url = `${BASE_URL}/api/v1/ws/market/${isin}?token=valid_token`;
  
  const response = ws.connect(url, {}, function (socket) {
    let messageCount = 0;
    let startTime = Date.now();
    
    // Connection established
    connectionSuccessRate.add(true);
    
    // Send initial subscription
    socket.send(JSON.stringify({
      type: 'subscribe',
      topic: `prices.${isin}`,
      subtopics: {
        levels: 'L1,L2',
        trades: true
      }
    }));
    
    // Handle incoming messages
    socket.on('message', function (message) {
      try {
        const data = JSON.parse(message);
        const latency = Date.now() - startTime;
        
        messagesReceived.add(1);
        messageLatency.add(latency);
        
        // Validate message format
        check(data, {
          'message has type': (msg) => msg.type !== undefined,
          'message has topic': (msg) => msg.topic !== undefined,
          'message has sequence': (msg) => msg.seq !== undefined,
          'message has timestamp': (msg) => msg.ts !== undefined,
        });
        
        messageCount++;
        startTime = Date.now();
        
        // Send ping every 30 seconds
        if (messageCount % 30 === 0) {
          socket.send(JSON.stringify({ type: 'ping' }));
        }
        
      } catch (e) {
        console.error('Error parsing message:', e);
      }
    });
    
    // Handle errors
    socket.on('error', function (error) {
      console.error('WebSocket error:', error);
      connectionSuccessRate.add(false);
    });
    
    // Handle close
    socket.on('close', function () {
      console.log('WebSocket connection closed');
    });
    
    // Keep connection alive for test duration
    setTimeout(function () {
      socket.close();
    }, TEST_DURATION * 1000);
  });
  
  check(response, {
    'WebSocket connection established': (r) => r && r.status === 101,
  });
}

// Auction test
export function auctionTest() {
  const auctionId = 'auction_123';
  const url = `${BASE_URL}/api/v1/ws/auction/${auctionId}?token=valid_token`;
  
  const response = ws.connect(url, {}, function (socket) {
    let messageCount = 0;
    let startTime = Date.now();
    
    connectionSuccessRate.add(true);
    
    socket.on('message', function (message) {
      try {
        const data = JSON.parse(message);
        const latency = Date.now() - startTime;
        
        messagesReceived.add(1);
        messageLatency.add(latency);
        
        messageCount++;
        startTime = Date.now();
        
        // Send ping every 30 seconds
        if (messageCount % 30 === 0) {
          socket.send(JSON.stringify({ type: 'ping' }));
        }
        
      } catch (e) {
        console.error('Error parsing message:', e);
      }
    });
    
    socket.on('error', function (error) {
      console.error('WebSocket error:', error);
      connectionSuccessRate.add(false);
    });
    
    setTimeout(function () {
      socket.close();
    }, TEST_DURATION * 1000);
  });
  
  check(response, {
    'WebSocket connection established': (r) => r && r.status === 101,
  });
}

// Trading test
export function tradingTest() {
  const userId = 'user_123';
  const url = `${BASE_URL}/api/v1/ws/trading/${userId}?token=valid_token`;
  
  const response = ws.connect(url, {}, function (socket) {
    let messageCount = 0;
    let startTime = Date.now();
    
    connectionSuccessRate.add(true);
    
    socket.on('message', function (message) {
      try {
        const data = JSON.parse(message);
        const latency = Date.now() - startTime;
        
        messagesReceived.add(1);
        messageLatency.add(latency);
        
        messageCount++;
        startTime = Date.now();
        
        // Send ping every 30 seconds
        if (messageCount % 30 === 0) {
          socket.send(JSON.stringify({ type: 'ping' }));
        }
        
      } catch (e) {
        console.error('Error parsing message:', e);
      }
    });
    
    socket.on('error', function (error) {
      console.error('WebSocket error:', error);
      connectionSuccessRate.add(false);
    });
    
    setTimeout(function () {
      socket.close();
    }, TEST_DURATION * 1000);
  });
  
  check(response, {
    'WebSocket connection established': (r) => r && r.status === 101,
  });
}

// Risk management test
export function riskTest() {
  const userId = 'user_123';
  const url = `${BASE_URL}/api/v1/ws/risk/${userId}?token=valid_token';
  
  const response = ws.connect(url, {}, function (socket) {
    let messageCount = 0;
    let startTime = Date.now();
    
    connectionSuccessRate.add(true);
    
    socket.on('message', function (message) {
      try {
        const data = JSON.parse(message);
        const latency = Date.now() - startTime;
        
        messagesReceived.add(1);
        messageLatency.add(latency);
        
        messageCount++;
        startTime = Date.now();
        
        // Send ping every 30 seconds
        if (messageCount % 30 === 0) {
          socket.send(JSON.stringify({ type: 'ping' }));
        }
        
      } catch (e) {
        console.error('Error parsing message:', e);
      }
    });
    
    socket.on('error', function (error) {
      console.error('WebSocket error:', error);
      connectionSuccessRate.add(false);
    });
    
    setTimeout(function () {
      socket.close();
    }, TEST_DURATION * 1000);
  });
  
  check(response, {
    'WebSocket connection established': (r) => r && r.status === 101,
  });
}

// Mobile test
export function mobileTest() {
  const userId = 'user_123';
  const url = `${BASE_URL}/api/v1/ws/mobile/${userId}?token=valid_token`;
  
  const response = ws.connect(url, {
    headers: {
      'x-device-type': 'mobile',
      'x-client-version': '1.0.0',
      'x-background-mode': 'false'
    }
  }, function (socket) {
    let messageCount = 0;
    let startTime = Date.now();
    
    connectionSuccessRate.add(true);
    
    socket.on('message', function (message) {
      try {
        const data = JSON.parse(message);
        const latency = Date.now() - startTime;
        
        messagesReceived.add(1);
        messageLatency.add(latency);
        
        messageCount++;
        startTime = Date.now();
        
        // Simulate background mode toggle
        if (messageCount % 60 === 0) {
          socket.send(JSON.stringify({
            type: 'background_mode',
            enabled: true
          }));
        } else if (messageCount % 90 === 0) {
          socket.send(JSON.stringify({
            type: 'background_mode',
            enabled: false
          }));
        }
        
        // Send ping every 30 seconds
        if (messageCount % 30 === 0) {
          socket.send(JSON.stringify({ type: 'ping' }));
        }
        
      } catch (e) {
        console.error('Error parsing message:', e);
      }
    });
    
    socket.on('error', function (error) {
      console.error('WebSocket error:', error);
      connectionSuccessRate.add(false);
    });
    
    setTimeout(function () {
      socket.close();
    }, TEST_DURATION * 1000);
  });
  
  check(response, {
    'WebSocket connection established': (r) => r && r.status === 101,
  });
}

// Reconnection test
export function reconnectionTest() {
  const isin = 'IN1234567890';
  const url = `${BASE_URL}/api/v1/ws/market/${isin}?token=valid_token`;
  
  let reconnectCount = 0;
  const maxReconnects = 3;
  
  function connect() {
    const response = ws.connect(url, {}, function (socket) {
      connectionSuccessRate.add(true);
      
      socket.on('message', function (message) {
        try {
          const data = JSON.parse(message);
          messagesReceived.add(1);
        } catch (e) {
          console.error('Error parsing message:', e);
        }
      });
      
      socket.on('error', function (error) {
        console.error('WebSocket error:', error);
        connectionSuccessRate.add(false);
      });
      
      socket.on('close', function () {
        console.log('WebSocket connection closed');
        
        // Attempt reconnection
        if (reconnectCount < maxReconnects) {
          reconnectCount++;
          reconnections.add(1);
          console.log(`Attempting reconnection ${reconnectCount}/${maxReconnects}`);
          
          setTimeout(connect, 1000 * reconnectCount); // Exponential backoff
        }
      });
      
      // Close connection after 30 seconds to test reconnection
      setTimeout(function () {
        socket.close();
      }, 30000);
    });
    
    check(response, {
      'WebSocket connection established': (r) => r && r.status === 101,
    });
  }
  
  connect();
}

// Main test execution
export default function () {
  // Mixed workload: 50% market data, 30% trading, 20% risk
  if (MIXED_WORKLOAD) {
    const rand = Math.random();
    if (rand < 0.5) {
      marketDataTest();
    } else if (rand < 0.8) {
      tradingTest();
    } else {
      riskTest();
    }
  } else {
    // Default: market data test
    marketDataTest();
  }
  
  // Add some reconnection testing
  if (Math.random() < 0.1) { // 10% chance
    reconnectionTest();
  }
}

// Setup and teardown
export function setup() {
  console.log('Starting WebSocket load test');
  console.log(`Base URL: ${BASE_URL}`);
  console.log(`Test duration: ${TEST_DURATION} seconds`);
  console.log(`Mixed workload: ${MIXED_WORKLOAD}`);
}

export function teardown(data) {
  console.log('WebSocket load test completed');
  console.log('Final metrics:');
  console.log(`- Connection success rate: ${connectionSuccessRate.values}`);
  console.log(`- Messages received: ${messagesReceived.values}`);
  console.log(`- Reconnections: ${reconnections.values}`);
}
