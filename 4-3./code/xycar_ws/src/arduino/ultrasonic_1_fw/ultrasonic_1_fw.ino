/*
HC-SR04 초음파 센서
*/

#define trig 2  // 트리거 핀 선언
#define echo 3  // 에코 핀 선언

void setup()
{
  Serial.begin(9600);     // 통신속도 9600bps로 시리얼 통신 시작
  // Serial.println("Start... Ultrasonic Sensor");
  pinMode(trig, OUTPUT);  // 트리거 핀을 출력으로 선언
  pinMode(echo, INPUT);   // 에코핀을 입력으로 선언
}

void loop() { 
  long duration, distance;  // 거리 측정을 위한 변수 선언
  // 트리거 핀으로 10us 동안 펄스 출력
  digitalWrite(trig, LOW);  // Trig 핀 Low
  delayMicroseconds(2);     // 2us 딜레이
  digitalWrite(trig, HIGH); // Trig 핀 High
  delayMicroseconds(10);    // 10us 딜레이
  digitalWrite(trig, LOW);  // Trig 핀 Low

  // pulseln() 함수는 핀에서 펄스신호를 읽어서 마이크로초 단위로 반환
  duration = pulseIn(echo, HIGH);
  distance = duration * 170 / 1000; // 왕복시간이므로 340m를 2로 나누어 170 곱하
  Serial.print("Distance(mm): ");
  Serial.println(distance); // 거리를 시리얼 모니터에 출력
  delay(100);
}
