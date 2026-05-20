/*
  Ray of HC-SR04 ultrasonic sensors.

  Each sensor is treated as one directional ray.
  The sensors can be placed around the robot at different angles.
*/

struct UltrasonicSensor {
  const char* name;
  int trigPin;
  int echoPin;
  float angleDeg;
  float distanceCm;
};


// Put your real GPIO pin numbers here.
// Replace 0 with the correct TRIG and ECHO pins for every sensor.
UltrasonicSensor sensors[] = {
  {
    "left",
    0,        // TODO: put LEFT sensor TRIG pin here
    0,        // TODO: put LEFT sensor ECHO pin here
    -45.0,
    999.0
  },
  {
    "front",
    0,        // TODO: put FRONT sensor TRIG pin here
    0,        // TODO: put FRONT sensor ECHO pin here
    0.0,
    999.0
  },
  {
    "right",
    0,        // TODO: put RIGHT sensor TRIG pin here
    0,        // TODO: put RIGHT sensor ECHO pin here
    45.0,
    999.0
  }
};


const int SENSOR_COUNT = sizeof(sensors) / sizeof(sensors[0]);

const float NO_DETECTION_DISTANCE_CM = 999.0;
const float OBSTACLE_THRESHOLD_CM = 20.0;


float readDistanceCm(UltrasonicSensor& sensor) {
  /*
    Reads distance from one HC-SR04 sensor.

    Steps:
    1. Set TRIG low.
    2. Send 10 microsecond pulse on TRIG.
    3. Measure ECHO pulse duration.
    4. Convert duration to distance in cm.
  */

  digitalWrite(sensor.trigPin, LOW);
  delayMicroseconds(2);

  digitalWrite(sensor.trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(sensor.trigPin, LOW);

  long duration = pulseIn(sensor.echoPin, HIGH, 30000);

  if (duration == 0) {
    return NO_DETECTION_DISTANCE_CM;
  }

  float distanceCm = duration * 0.0343 / 2.0;

  return distanceCm;
}


void setupSensors() {
  /*
    Sets TRIG pins as outputs and ECHO pins as inputs.
  */

  for (int i = 0; i < SENSOR_COUNT; i++) {
    pinMode(sensors[i].trigPin, OUTPUT);
    pinMode(sensors[i].echoPin, INPUT);

    digitalWrite(sensors[i].trigPin, LOW);
  }
}


void updateSensorRay() {
  /*
    Updates distance value for every sensor in the ray.
  */

  for (int i = 0; i < SENSOR_COUNT; i++) {
    sensors[i].distanceCm = readDistanceCm(sensors[i]);

    // Small delay between sensors to avoid interference
    delay(60);
  }
}


int getClosestSensorIndex() {
  /*
    Returns the index of the sensor that detected the closest object.
  */

  int closestIndex = 0;

  for (int i = 1; i < SENSOR_COUNT; i++) {
    if (sensors[i].distanceCm < sensors[closestIndex].distanceCm) {
      closestIndex = i;
    }
  }

  return closestIndex;
}


float getClosestDistance() {
  /*
    Returns the closest detected distance.
  */

  int closestIndex = getClosestSensorIndex();

  return sensors[closestIndex].distanceCm;
}


bool isObstacleDetected() {
  /*
    Checks if any sensor detected an obstacle closer than the threshold.
  */

  return getClosestDistance() < OBSTACLE_THRESHOLD_CM;
}


void printSensorRay() {
  /*
    Prints current readings in the Serial Monitor.
  */

  Serial.println("=== Sensor ray readings ===");

  for (int i = 0; i < SENSOR_COUNT; i++) {
    Serial.print(sensors[i].name);
    Serial.print(" | angle: ");
    Serial.print(sensors[i].angleDeg);
    Serial.print(" deg | distance: ");
    Serial.print(sensors[i].distanceCm);
    Serial.println(" cm");
  }

  int closestIndex = getClosestSensorIndex();

  Serial.print("Closest obstacle: ");
  Serial.print(sensors[closestIndex].name);
  Serial.print(" | ");
  Serial.print(sensors[closestIndex].distanceCm);
  Serial.println(" cm");

  if (isObstacleDetected()) {
    Serial.println("Obstacle detected!");
  } else {
    Serial.println("No obstacle nearby.");
  }

  Serial.println();
}


void setup() {
  Serial.begin(115200);

  setupSensors();
}


void loop() {
  updateSensorRay();

  printSensorRay();

  delay(200);
}