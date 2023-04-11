#include "wiring_private.h"
#include <math.h>

#define MAX_DISTANCE 201*58.2 
#define PING_INTERVAL 33 

//low pass filter 클라스 정의
#ifndef LPF_h
#define LPF_h
class LPF
{
public:
  float alpha;
  float prev_z;
  
  LPF(float al)
  {
    alpha = al;
  }

  float LPF::Estimate(float z, int i)
  {
    //prev_z = alpha * prev_z + (1 - alpha) * z;
    prev_z = (i-1) / i * prev_z + z / i;
    return prev_z;
  }
};
#endif
//low pass filter 클라스 정의 끝

//low pass filter
float alpha = 0.9;
LPF lpf[4] = {LPF(alpha), LPF(alpha), LPF(alpha), LPF(alpha)};

int trig[4] = {2, 4, 6, 8};
int echo[4] = {3, 5, 7, 9};

unsigned long pingTimer;
long around_time[4]; // change
long distance_cm[4]; // change
float measured_value[4]; // change
float estimated_value[4]; // change
int num;
int num_sensor = 1;
int x = 0;
float last_estimate = 0.0;
int a = 1;

unsigned long pulseIn(uint8_t pin, uint8_t state, unsigned long timeout)
{
  uint8_t bit = digitalPinToBitMask(pin);
  uint8_t port = digitalPinToPort(pin);
  uint8_t stateMask = (state ? bit : 0);

  unsigned long maxloops = microsecondsToClockCycles(timeout)/16;

  unsigned long width = countPulseASM(portInputRegister(port), bit, stateMask, maxloops);

  if (width)
    return clockCyclesToMicroseconds(width * 16 + 16);
  else
    return MAX_DISTANCE;
}

void setup() 
{
  Serial.begin(115200);

  for (uint8_t i=0; i<num_sensor; i++) // change
  {
    pinMode(trig[i], OUTPUT);
    pinMode(echo[i], INPUT);
    Serial.println(trig[i]);
    Serial.println(echo[i]);
  }

  pingTimer = millis();
}

void loop() 
{
  if (millis() >= pingTimer) 
  {   
    pingTimer += PING_INTERVAL;
    around_time[num] = trig_ultra(trig[num], echo[num]);
    num++;      
  }
  if (num>3) // change
  { 
    num=0; 
    oneSensorCycle();   
    pingTimer = millis();
  }

  for(int i=0; i<num_sensor; i++) { // change
    float rand_num = (1 + random(-100,100) / 500.0);
    measured_value[i] = distance_cm[i] * rand_num;
    estimated_value[i] = lpf[i].Estimate(measured_value[i], a);
  }
  a++;
  int print_num = 0;
  Serial.print(',');
  Serial.print("measure");
  Serial.print(",");
  Serial.print(measured_value[print_num]);
  Serial.print(",");
  Serial.print("esti");
  Serial.print(",");
  Serial.print(estimated_value[print_num]);
  Serial.println(",");

}

long trig_ultra(int TRIG,int ECHO)
{
  long receive_time_ms;

  digitalWrite(TRIG, LOW); 
  delayMicroseconds(2); 
  digitalWrite(TRIG, HIGH); 
  delayMicroseconds(10);
  digitalWrite(TRIG, LOW);

  receive_time_ms = pulseIn(ECHO, HIGH, MAX_DISTANCE);

  return(receive_time_ms);
}

void oneSensorCycle() 
{ 
  for (uint8_t i=0; i<num_sensor; i++) 
  {
      //if(i==0) {
        distance_cm[i] = around_time[i] / 58.2;
        //Serial.print(distance_cm[i]);
        //Serial.print(i+1);
        //Serial.println();
      //}
  }
}
