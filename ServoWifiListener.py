#include <WiFi.h>
#include <ESP32Servo.h>
#include <ESPmDNS.h>

Servo myServo;

// Servo control variables
float currentPos = 90.0;
int targetPos = 90;
unsigned long lastUpdateTime = 0;
const int updateInterval = 5;  // 10ms = 100Hz update rate
const float smoothingFactor = 0.3;

// WiFi credentials
const char* ssid = "Oliver_iPhone";
const char* password = "password";

// Network
WiFiServer server(12345);
WiFiClient client;

// Command buffer (using char array instead of String for efficiency)
char commandBuffer[8];
uint8_t bufferIndex = 0;

// WiFi watchdog
unsigned long lastWiFiCheck = 0;
const unsigned long wifiCheckInterval = 5000;  // Check every 5 seconds

void connectToWiFi() {
  Serial.print("🔹 Connecting to Wi-Fi: ");
  Serial.println(ssid);
  
  WiFi.mode(WIFI_STA);  // Explicitly set station mode
  WiFi.begin(ssid, password);
  
  int retries = 0;
  while (WiFi.status() != WL_CONNECTED && retries < 30) {
    delay(500);
    Serial.print(".");
    retries++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✅ Connected to Wi-Fi!");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\n❌ Failed to connect to Wi-Fi. Restarting...");
    delay(1000);
    ESP.restart();
  }
}

void checkWiFiConnection() {
  unsigned long currentTime = millis();
  if (currentTime - lastWiFiCheck >= wifiCheckInterval) {
    lastWiFiCheck = currentTime;
    
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("⚠️ WiFi disconnected! Reconnecting...");
      connectToWiFi();
      
      // Restart server after reconnection
      server.begin();
    }
  }
}

void updateServoPosition() {
  unsigned long currentTime = millis();
  
  if (currentTime - lastUpdateTime >= updateInterval) {
    lastUpdateTime = currentTime;
    
    float error = targetPos - currentPos;
    
    // Only update if error is significant
    if (abs(error) > 0.5) {
      currentPos += error * smoothingFactor;
      int writePos = (int)round(currentPos);
      
      // Clamp to valid range
      writePos = constrain(writePos, 0, 180);
      myServo.write(writePos);
    } else {
      currentPos = targetPos;
    }
  }
}

void processCommand() {
  // Non-blocking read of available bytes
  while (client.available() > 0) {
    char c = client.read();
    
    if (c == '\n' || c == '\r') {
      // End of command
      if (bufferIndex > 0) {
        commandBuffer[bufferIndex] = '\0';  // Null terminate
        
        // Parse angle
        int newTarget = atoi(commandBuffer);
        
        if (newTarget >= 0 && newTarget <= 180) {
          targetPos = newTarget;
          // Minimal serial output to reduce latency
          // Serial.printf("→ %d°\n", targetPos);
        }
        
        // Reset buffer
        bufferIndex = 0;
      }
    } 
    else if (c >= '0' && c <= '9') {
      // Add digit to buffer (with overflow protection)
      if (bufferIndex < sizeof(commandBuffer) - 1) {
        commandBuffer[bufferIndex++] = c;
      }
    }
    // Ignore other characters
  }
}

void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println("\n🔹 ESP32 Servo Controller Starting...");
  
  // Initialize servo
  myServo.attach(13);
  myServo.write((int)currentPos);
  Serial.println("✅ Servo initialized at 90°");
  delay(500);
  
  // Connect to WiFi
  connectToWiFi();
  
  // Start mDNS
  if (!MDNS.begin("esp32")) {
    Serial.println("❌ mDNS failed to start");
  } else {
    Serial.println("✅ mDNS started: esp32.local");
  }
  
  // Start TCP server
  server.begin();
  server.setNoDelay(true);  // Disable Nagle's algorithm for lower latency
  Serial.println("✅ Server started on port 12345");
  
  Serial.println("🔹 Ready for connections!");
  
  lastUpdateTime = millis();
  lastWiFiCheck = millis();
  
  // Initialize command buffer
  bufferIndex = 0;
}

void loop() {
  // CRITICAL: Always update servo position first
  updateServoPosition();
  
  // Check WiFi connection periodically
  checkWiFiConnection();
  
  // Handle client connection (non-blocking)
  if (!client || !client.connected()) {
    // Try to accept new client
    client = server.accept();
    
    if (client) {
      Serial.println("🔹 Client connected");
      client.setNoDelay(true);  // Disable Nagle's algorithm
      bufferIndex = 0;  // Reset command buffer
    }
  } else {
    // Client is connected, process any incoming commands
    processCommand();
  }
  
  // Yield to prevent watchdog timer issues
  yield();
}