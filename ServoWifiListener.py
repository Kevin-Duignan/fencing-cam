#include <WiFi.h>
#include <WiFiManager.h>
#include <ESP32Servo.h>
#include <ESPmDNS.h>

Servo myServo;
int currentPos = 90;         
int targetPos = 90;          
int stepDelay = 10;          
int stepSize = 1;            

// Use port 12345 to match Python
WiFiServer server(12345);

void setup() {
  Serial.begin(115200);
  delay(100);

  Serial.println("🔹 Setup start");

  // Servo startup
  myServo.attach(13);       // Change to your chosen GPIO
  myServo.write(currentPos);

  // WiFiManager
  WiFiManager wm;
  Serial.println("🔹 WiFiManager object created");
  // wm.resetSettings();  // Uncomment to reset credentials

  if (!wm.autoConnect("ESP32_AP", "12345678")) {
    Serial.println("❌ Failed to connect, restarting...");
    ESP.restart();
  }

  Serial.println("✅ WiFi connected!");
  Serial.print("SSID: "); Serial.println(WiFi.SSID());
  Serial.print("IP: "); Serial.println(WiFi.localIP());

  // mDNS
  if (!MDNS.begin("esp32")) {
    Serial.println("❌ mDNS failed!");
  } else {
    Serial.println("✅ mDNS started: esp32.local");
  }

  server.begin();
  Serial.println("🔹 Server started on port 12345");
  Serial.println("🔹 Setup complete");
}

void loop() {
  WiFiClient client = server.accept();
  if (client) {
    Serial.println("🔹 Client connected");

    while (client.connected()) {
      if (client.available()) {
        String line = client.readStringUntil('\n');
        line.trim();
        int newTarget = line.toInt();
        if (newTarget >= 0 && newTarget <= 180) {
          targetPos = newTarget;
          Serial.print("Received angle: "); Serial.println(targetPos);
        }
      }

      // Smooth servo motion
      if (currentPos < targetPos) {
        currentPos += stepSize;
        if (currentPos > targetPos) currentPos = targetPos;
        myServo.write(currentPos);
        delay(stepDelay);
      } 
      else if (currentPos > targetPos) {
        currentPos -= stepSize;
        if (currentPos < targetPos) currentPos = targetPos;
        myServo.write(currentPos);
        delay(stepDelay);
      }
    }

    client.stop();
    Serial.println("🔹 Client disconnected");
  }
}