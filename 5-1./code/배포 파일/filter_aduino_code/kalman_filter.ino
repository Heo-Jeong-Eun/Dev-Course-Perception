#include "wiring_private.h"
#include <math.h>
#include <ArduinoQueue.h>

#define MAX_DISTANCE 201*58.2 
#define PING_INTERVAL 33 

#ifndef SimpleKalmanFilter_h
#define SimpleKalmanFilter_h

class KalmanFilter
{
public:
  KalmanFilter(float init_est, float init_est_e, float mea_e, float proc_e);
  float updateEstimate(float mea, float past_value);
  float _last_estimate;

private:
  float _err_measure;
  float _err_estimate;
  float _err_processing;
  float _current_estimate;
  float _kalman_gain;
};

#endif
KalmanFilter::KalmanFilter(float init_est, float init_est_e, float mea_e, float proc_e)
{
  _last_estimate = init_est;
  _err_estimate=init_est_e;
  _err_measure=mea_e;
  _err_processing=proc_e;
}

float KalmanFilter::updateEstimate(float mea, float past_value)
{
  _last_estimate =  (_last_estimate + past_value) / 2 + sqrt(_err_estimate) * random(-100, 100) / 100;
  _err_estimate = _err_estimate + _err_processing;
  _kalman_gain = _err_estimate/(_err_estimate + _err_measure);
  _current_estimate = _last_estimate + _kalman_gain * (mea - _last_estimate);
  _err_estimate =  (1.0 - _kalman_gain)*_err_estimate;
  _last_estimate=_current_estimate;

  return _current_estimate;
}

int timestep_for_filter = 10;
float init_estimate = 0.0;
float init_estimate_noise = 5.0;
float measure_noise = 5.0;
float process_noise = 0.1;

KalmanFilter kalmanFilter[4] = 
{ KalmanFilter(init_estimate, init_estimate_noise, measure_noise, process_noise), 
  KalmanFilter(init_estimate, init_estimate_noise, measure_noise, process_noise), 
  KalmanFilter(init_estimate, init_estimate_noise, measure_noise, process_noise), 
  KalmanFilter(init_estimate, init_estimate_noise, measure_noise, process_noise)};
  
ArduinoQueue<float> prev_mea(timestep_for_filter);

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
float past_value = 0.0;

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

  // 측정 값을 받아 현재 상태 예측
  for(int i=0; i<num_sensor; i++) {
    float rand_num = (1 + random(-100,100) / 500.0);
    
    measured_value[i] = distance_cm[i] * rand_num;
    prev_mea.enqueue(measured_value[i]);
    if(prev_mea.isFull()){
      past_value = prev_mea.dequeue();
    }else{
      past_value = 0.0;
    }
    estimated_value[i] = kalmanFilter[i].updateEstimate(measured_value[i], past_value);
  }
  
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
